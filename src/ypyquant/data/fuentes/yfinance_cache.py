"""
Adaptador de producción de la fuente de precios: yfinance con caché Parquet por valor.

Responsabilidad:
- `FuenteYFinanceCache`: guarda cada símbolo en `data/cache_precios/<símbolo>.parquet` (+ un .json con el
  rango cubierto) y solo llama a yfinance si el rango pedido no está cubierto. Sin red devuelve lo que haya en
  caché (con aviso) o None. `YPYQUANT_OFFLINE=1` desactiva las descargas. Los precios crudos
  (`ajuste="crudo"`, `auto_adjust=False`) van a un segundo par de archivos `<símbolo>.crudo.*`; la serie
  ajustada se guarda como siempre, así que la caché existente sigue siendo válida. Nota: Yahoo ya ajusta
  `Close` por splits, de modo que su "crudo" solo es crudo respecto a los dividendos.
- Deslistados: yfinance no los sabe. `deslistado()` mantiene la inferencia de siempre como comportamiento
  del adaptador: si el último dato de la caché queda más de `DIAS_PRECIO_CADUCADO` antes del final del rango
  cubierto, el valor dejó de cotizar (`Deslistado.inferido=True`, sin motivo).
- Identificador: yfinance no ofrece uno estable; se usa el propio símbolo (`yf:<símbolo>`).

Formato de salida (como espera `BacktestEngine`): índice fecha (DatetimeIndex ordenado, sin zona),
columnas `Open, High, Low, Close, Volume, adjusted_close` (en ajustado, Close ya incluye splits y dividendos).

Se relaciona con:
- ypyquant.data.fuente_precios (interfaz, `inferir_deslistado`) y ypyquant.data.fuentes (`fuente_por_defecto`).
- ypyquant.data.universos (formato de símbolos: el canónico es el de yfinance, p. ej. BRK-B).
- tests/test_fuente_precios_contrato.py (corre sin red con la caché precargada) y tests/test_estrategias.py.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ypyquant.data.fuente_precios import Ajuste, Deslistado, FuentePrecios, inferir_deslistado

logger = logging.getLogger(__name__)

RAIZ_REPO = Path(__file__).resolve().parents[4]
CARPETA_CACHE_POR_DEFECTO = RAIZ_REPO / "data" / "cache_precios"


def _nombre_archivo(simbolo: str) -> str:
    """Nombre de archivo seguro para un símbolo ('^GSPC' -> '_GSPC', 'BRK.B' -> 'BRK.B')."""
    return "".join(c if c.isalnum() or c in ".-" else "_" for c in simbolo)


def _normalizar_ohlcv(crudo: pd.DataFrame, simbolo: str) -> pd.DataFrame | None:
    """Deja el DataFrame de yfinance en el formato estándar (columnas simples, índice sin zona, adjusted_close)."""
    if crudo is None or crudo.empty:
        return None
    df = crudo.copy()
    if isinstance(df.columns, pd.MultiIndex):
        nivel = df.columns.get_level_values(-1)
        df.columns = df.columns.get_level_values(0) if simbolo in nivel else nivel
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if "Close" not in df.columns:
        return None
    df = df.dropna(subset=["Close"])
    df["adjusted_close"] = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    columnas = [c for c in ("Open", "High", "Low", "Close", "Volume", "adjusted_close") if c in df.columns]
    return df[columnas]


class FuenteYFinanceCache(FuentePrecios):
    """
    Fuente yfinance con caché en Parquet por símbolo.

    Uso: `FuenteYFinanceCache().ohlcv('AAPL', '2018-01-01', '2024-12-31')` (o llamarla como cargador). Es
    segura entre hilos para símbolos distintos.
    """

    nombre = "yfinance+cache"

    def __init__(self, carpeta: Path | str | None = None, descargar: bool | None = None):
        self.carpeta = Path(carpeta) if carpeta else CARPETA_CACHE_POR_DEFECTO
        self.descargar = (os.environ.get("YPYQUANT_OFFLINE") != "1") if descargar is None else descargar

    # -- caché ----------------------------------------------------------------

    def _rutas(self, simbolo: str, ajuste: Ajuste) -> tuple[Path, Path]:
        base = self.carpeta / (_nombre_archivo(simbolo) + (".crudo" if ajuste == "crudo" else ""))
        return Path(f"{base}.parquet"), Path(f"{base}.json")

    def _leer_cache(self, simbolo: str, ajuste: Ajuste = "ajustado") -> tuple[pd.DataFrame | None, tuple[str, str] | None]:
        ruta_datos, ruta_meta = self._rutas(simbolo, ajuste)
        if not ruta_datos.exists() or not ruta_meta.exists():
            return None, None
        try:
            meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
            return pd.read_parquet(ruta_datos), (meta["inicio"], meta["fin"])
        except Exception as error:  # noqa: BLE001 - caché corrupta: se vuelve a descargar
            logger.warning("Caché de %s ilegible (%s); se ignora", simbolo, error)
            return None, None

    def _guardar_cache(self, simbolo: str, df: pd.DataFrame, inicio: str, fin: str, ajuste: Ajuste = "ajustado") -> None:
        ruta_datos, ruta_meta = self._rutas(simbolo, ajuste)
        ruta_datos.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ruta_datos)
        ruta_meta.write_text(json.dumps({"inicio": inicio, "fin": fin, "actualizado": datetime.now().isoformat()}),
                             encoding="utf-8")

    def _descargar(self, simbolo: str, inicio: str, fin: str, ajuste: Ajuste = "ajustado") -> pd.DataFrame | None:
        import yfinance as yf

        fin_exclusivo = (pd.Timestamp(fin) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            crudo = yf.download(simbolo, start=inicio, end=fin_exclusivo, auto_adjust=(ajuste == "ajustado"),
                                progress=False, threads=False)
        except Exception as error:  # noqa: BLE001 - yfinance lanza de todo; sin red se usa la caché
            logger.warning("Descarga de %s falló: %s", simbolo, error)
            return None
        return _normalizar_ohlcv(crudo, simbolo)

    # -- interfaz de FuentePrecios -------------------------------------------

    def ohlcv(self, simbolo: str, fecha_inicio: str, fecha_fin: str, ajuste: Ajuste = "ajustado") -> pd.DataFrame | None:
        hoy = date.today().strftime("%Y-%m-%d")
        fin_pedido = min(fecha_fin, hoy)
        cache, cubierto = self._leer_cache(simbolo, ajuste)
        if cache is not None and cubierto and cubierto[0] <= fecha_inicio and cubierto[1] >= fin_pedido:
            return cache.loc[fecha_inicio:fecha_fin]
        if self.descargar:
            inicio_total = min(fecha_inicio, cubierto[0]) if cubierto else fecha_inicio
            fin_total = max(fin_pedido, cubierto[1]) if cubierto else fin_pedido
            nuevo = self._descargar(simbolo, inicio_total, fin_total, ajuste)
            if nuevo is not None and not nuevo.empty:
                self._guardar_cache(simbolo, nuevo, inicio_total, fin_total, ajuste)
                return nuevo.loc[fecha_inicio:fecha_fin]
        if cache is not None:
            logger.warning("%s: sin red o sin datos nuevos; se usa la caché parcial (%s a %s)", simbolo, *cubierto)
            return cache.loc[fecha_inicio:fecha_fin]
        return None

    def identificador(self, simbolo: str) -> str:
        return f"yf:{simbolo}"

    def deslistado(self, simbolo: str) -> Deslistado | None:
        """Inferido de la caché ajustada: último dato frente al final del rango que se ha pedido (sin red)."""
        cache, cubierto = self._leer_cache(simbolo)
        if cache is None or cache.empty or not cubierto:
            return None
        return inferir_deslistado(cache.index[-1], pd.Timestamp(cubierto[1]))
