"""
Cargadores de precios para experimentos: caché en Parquet sobre yfinance y generador sintético.

Responsabilidad:
- `CargadorPreciosCache`: cumple el contrato `CargadorPrecios` `(ticker, inicio, fin) -> OHLCV`.
  Guarda cada ticker en `data/cache_precios/<ticker>.parquet` (+ un .json con el rango cubierto) y
  solo llama a yfinance si el rango pedido no está cubierto. Sin red devuelve lo que haya en
  caché (con aviso) o None. `YPYQUANT_OFFLINE=1` desactiva las descargas.
- `CargadorSintetico`: precios deterministas (paseo geométrico con semilla por ticker) sin red;
  sirve para tests y para probar la UI sin datos (`YPYQUANT_PRECIOS=sintetico`).
- `cargador_por_defecto()`: elige uno según el entorno.

Formato de salida (como espera `BacktestEngine`): índice fecha (DatetimeIndex ordenado, sin zona),
columnas `Open, High, Low, Close, Volume, adjusted_close` (Close ya ajustado por splits y dividendos).

Se relaciona con:
- ypyquant.data.matriz_precios (contrato CargadorPrecios).
- ypyquant.estrategias.experimentos (los usa para cargar universo y benchmark).
- app/api (los inyecta; los tests usan el sintético).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ypyquant.data.matriz_precios import CargadorPrecios

logger = logging.getLogger(__name__)

RAIZ_REPO = Path(__file__).resolve().parents[3]
CARPETA_CACHE_POR_DEFECTO = RAIZ_REPO / "data" / "cache_precios"
TOLERANCIA_DIAS_FIN = 5  # fines de semana y festivos: el último dato puede quedar unos días antes de `fin`


def _nombre_archivo(ticker: str) -> str:
    """Nombre de archivo seguro para un ticker ('^GSPC' -> '_GSPC', 'BRK.B' -> 'BRK.B')."""
    return "".join(c if c.isalnum() or c in ".-" else "_" for c in ticker)


def _normalizar_ohlcv(crudo: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Deja el DataFrame de yfinance en el formato estándar (columnas simples, índice sin zona, adjusted_close)."""
    if crudo is None or crudo.empty:
        return None
    df = crudo.copy()
    if isinstance(df.columns, pd.MultiIndex):
        nivel = df.columns.get_level_values(-1)
        df.columns = df.columns.get_level_values(0) if ticker in nivel else nivel
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if "Close" not in df.columns:
        return None
    df = df.dropna(subset=["Close"])
    df["adjusted_close"] = df["Close"]
    columnas = [c for c in ("Open", "High", "Low", "Close", "Volume", "adjusted_close") if c in df.columns]
    return df[columnas]


class CargadorPreciosCache:
    """
    Cargador yfinance con caché en Parquet por ticker.

    Uso: `CargadorPreciosCache()('AAPL', '2018-01-01', '2024-12-31')`. Es seguro entre hilos
    para tickers distintos; para el mismo ticker usa un cerrojo por instancia.
    """

    def __init__(self, carpeta: Path | str | None = None, descargar: bool | None = None):
        self.carpeta = Path(carpeta) if carpeta else CARPETA_CACHE_POR_DEFECTO
        self.descargar = (os.environ.get("YPYQUANT_OFFLINE") != "1") if descargar is None else descargar

    def _rutas(self, ticker: str) -> tuple[Path, Path]:
        base = self.carpeta / _nombre_archivo(ticker)
        return base.with_suffix(".parquet"), base.with_suffix(".json")

    def _leer_cache(self, ticker: str) -> tuple[pd.DataFrame | None, tuple[str, str] | None]:
        ruta_datos, ruta_meta = self._rutas(ticker)
        if not ruta_datos.exists() or not ruta_meta.exists():
            return None, None
        try:
            meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
            return pd.read_parquet(ruta_datos), (meta["inicio"], meta["fin"])
        except Exception as error:  # noqa: BLE001 - caché corrupta: se vuelve a descargar
            logger.warning("Caché de %s ilegible (%s); se ignora", ticker, error)
            return None, None

    def _guardar_cache(self, ticker: str, df: pd.DataFrame, inicio: str, fin: str) -> None:
        ruta_datos, ruta_meta = self._rutas(ticker)
        ruta_datos.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ruta_datos)
        ruta_meta.write_text(json.dumps({"inicio": inicio, "fin": fin, "actualizado": datetime.now().isoformat()}),
                             encoding="utf-8")

    def _descargar(self, ticker: str, inicio: str, fin: str) -> pd.DataFrame | None:
        import yfinance as yf

        fin_exclusivo = (pd.Timestamp(fin) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            crudo = yf.download(ticker, start=inicio, end=fin_exclusivo, auto_adjust=True, progress=False,
                                threads=False)
        except Exception as error:  # noqa: BLE001 - yfinance lanza de todo; sin red se usa la caché
            logger.warning("Descarga de %s falló: %s", ticker, error)
            return None
        return _normalizar_ohlcv(crudo, ticker)

    def __call__(self, ticker: str, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame | None:
        hoy = date.today().strftime("%Y-%m-%d")
        fin_pedido = min(fecha_fin, hoy)
        cache, cubierto = self._leer_cache(ticker)
        if cache is not None and cubierto and cubierto[0] <= fecha_inicio and cubierto[1] >= fin_pedido:
            return cache.loc[fecha_inicio:fecha_fin]
        if self.descargar:
            inicio_total = min(fecha_inicio, cubierto[0]) if cubierto else fecha_inicio
            fin_total = max(fin_pedido, cubierto[1]) if cubierto else fin_pedido
            nuevo = self._descargar(ticker, inicio_total, fin_total)
            if nuevo is not None and not nuevo.empty:
                self._guardar_cache(ticker, nuevo, inicio_total, fin_total)
                return nuevo.loc[fecha_inicio:fecha_fin]
        if cache is not None:
            logger.warning("%s: sin red o sin datos nuevos; se usa la caché parcial (%s a %s)", ticker, *cubierto)
            return cache.loc[fecha_inicio:fecha_fin]
        return None


class CargadorSintetico:
    """
    Precios sintéticos deterministas: paseo geométrico con deriva y volatilidad propias por ticker.

    La semilla depende solo del ticker (hash estable), así que los mismos datos salen siempre.
    Días hábiles de lunes a viernes. El benchmark sintético 'SPY' se genera igual que cualquier ticker.
    """

    def __init__(self, deriva_anual: float = 0.08, volatilidad_anual: float = 0.22):
        self.deriva_anual = deriva_anual
        self.volatilidad_anual = volatilidad_anual

    def __call__(self, ticker: str, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
        semilla = int(hashlib.sha256(ticker.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(semilla)
        fechas = pd.bdate_range("2010-01-04", max(pd.Timestamp(fecha_fin), pd.Timestamp("2010-01-05")))
        deriva = self.deriva_anual * (0.5 + rng.random())
        vol = self.volatilidad_anual * (0.6 + 0.8 * rng.random())
        retornos = rng.normal(deriva / 252, vol / np.sqrt(252), len(fechas))
        cierre = 100 * np.exp(np.cumsum(retornos))
        df = pd.DataFrame({
            "Open": cierre * (1 + rng.normal(0, 0.002, len(fechas))),
            "High": cierre * 1.005, "Low": cierre * 0.995, "Close": cierre,
            "Volume": rng.integers(1_000_000, 50_000_000, len(fechas)).astype(float),
            "adjusted_close": cierre,
        }, index=fechas)
        return df.loc[fecha_inicio:fecha_fin]


def cargador_por_defecto() -> CargadorPrecios:
    """`YPYQUANT_PRECIOS=sintetico` da el sintético; en otro caso, yfinance con caché en Parquet."""
    if os.environ.get("YPYQUANT_PRECIOS", "").lower() == "sintetico":
        return CargadorSintetico()
    return CargadorPreciosCache()
