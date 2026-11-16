"""
Descarga factores Fama-French 5 + Momentum diarios desde Kenneth French Data Library.

Flujo principal (FrenchDataFetcher.get_ff5_factors):
  Cache en disco (Parquet, TTL 7 días) → si caduca, descarga ZIP con retry (3 intentos)
  → parsea CSV French → combina FF5 + MOM → filtra por fechas → cache en memoria.

Columnas de salida: MKT, SMB, HML, RMW, CMA, MOM, RF (decimal: 0.01 = 1 %).

Relacionado con: factor_models.py (FamaFrenchModel.fit() consume estos datos),
walk_forward.py (extrae RF de la columna RF).
"""

import io
import time
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# URLs de descarga directa desde Kenneth French Data Library
URLS = {
    'ff5_daily': (
        'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
        'F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    ),
    'momentum_daily': (
        'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
        'F-F_Momentum_Factor_daily_CSV.zip'
    ),
}

# Directorio de cache persistente en disco (Parquet, TTL = 7 días)
_CACHE_DIR = Path.home() / ".cache" / "quant" / "ff_data"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL_DIAS = 7


def _cache_path(clave: str) -> Path:
    """Devuelve la ruta del archivo Parquet para una clave de cache."""
    return _CACHE_DIR / f"{clave}.parquet"


def _cache_valida(clave: str) -> bool:
    """True si el archivo Parquet existe y tiene menos de _CACHE_TTL_DIAS días."""
    ruta = _cache_path(clave)
    if not ruta.exists():
        return False
    edad = datetime.now() - datetime.fromtimestamp(ruta.stat().st_mtime)
    return edad < timedelta(days=_CACHE_TTL_DIAS)


def _descargar_con_retry(url: str, intentos: int = 3, espera: float = 2.0) -> bytes:
    """
    Descarga una URL con reintentos exponenciales simples.

    I: url, intentos (default 3), espera en segundos entre reintentos (default 2s).
    T: urlopen con timeout=30; espera fija entre fallos; lanza la última excepción si agota.
    O: bytes — contenido bruto de la respuesta.
    """
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    ultimo_exc: Exception = RuntimeError("Sin intentos")
    for i in range(intentos):
        try:
            respuesta = urllib.request.urlopen(req, timeout=30)
            return respuesta.read()
        except Exception as exc:
            ultimo_exc = exc
            if i < intentos - 1:
                logger.warning(
                    f"[FrenchDataFetcher] Intento {i + 1}/{intentos} fallido: {exc}. "
                    f"Reintentando en {espera}s..."
                )
                time.sleep(espera)
    raise ultimo_exc


def _descargar_csv_desde_zip(url: str) -> str:
    """
    Descarga ZIP de Kenneth French (con retry) y extrae el primer CSV.

    I: URL del archivo ZIP.
    T: _descargar_con_retry → ZipFile → decode UTF-8.
    O: str con el contenido del CSV.
    """
    datos_zip = _descargar_con_retry(url)

    with zipfile.ZipFile(io.BytesIO(datos_zip)) as z:
        nombre_csv = z.namelist()[0]
        return z.read(nombre_csv).decode('utf-8')


def _parsear_csv_french(contenido_csv: str) -> pd.DataFrame:
    """
    Parsea el formato CSV de Kenneth French Data Library.

    I: str con el contenido del CSV (cabecera con coma, valores en porcentaje).
    T: Localiza la línea de cabecera → pd.read_csv → elimina filas no numéricas
       → índice YYYYMMDD → DatetimeIndex → divide por 100 para convertir a decimal.
    O: DataFrame con índice de fechas y factores en decimal (0.01 = 1 %).
    """
    lineas = contenido_csv.split('\n')

    # Localizar la línea de cabecera (primera que empieza con coma)
    indice_cabecera = None
    for i, linea in enumerate(lineas):
        if linea.strip().startswith(','):
            indice_cabecera = i
            break

    if indice_cabecera is None:
        raise ValueError("No se encontró cabecera en el CSV de French Data Library")

    # Leer CSV desde la cabecera
    df = pd.read_csv(
        io.StringIO('\n'.join(lineas[indice_cabecera:])),
        index_col=0,
        skipinitialspace=True,
    )

    # Limpiar nombres de columnas (pueden tener espacios extra)
    df.columns = df.columns.str.strip()

    # Eliminar filas no numéricas (copyright al final, líneas vacías)
    df = df[pd.to_numeric(df.index, errors='coerce').notna()]

    # Índice YYYYMMDD → DatetimeIndex
    df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d')
    df.index.name = 'date'

    # Porcentaje → decimal
    df = df.astype(float) / 100.0

    return df


class FrenchDataFetcher:
    """
    Descarga factores Fama-French 5 + Momentum diarios
    directamente desde Kenneth French Data Library (ZIP → CSV).

    Cache de dos niveles:
    1. En memoria (dict en instancia) — evita re-lecturas durante la sesión.
    2. En disco (Parquet en ~/.cache/quant/ff_data/) con TTL de 7 días — evita
       descargas repetidas entre sesiones. La clave de disco es siempre 'ff5mom_all'
       (datos completos sin filtro) y el filtrado por fechas se hace en memoria.
    """

    def __init__(self):
        self._cache: dict = {}

    def get_ff5_factors(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Descarga Fama-French 5 factores + Momentum diarios.

        La serie completa se cachea en disco (Parquet, TTL 7 días) para evitar
        descargas en cada sesión. El filtrado por fechas se aplica tras la carga.

        Args:
            start_date: Fecha inicio YYYY-MM-DD (opcional, sin filtro si None)
            end_date: Fecha fin YYYY-MM-DD (opcional, sin filtro si None)

        Returns:
            DataFrame con columnas ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'RF']
            en formato decimal (0.01 = 1%)
        """
        clave_sesion = f"ff5mom_{start_date}_{end_date}"
        if clave_sesion in self._cache:
            return self._cache[clave_sesion]

        # ── Cache en disco (datos completos, sin filtro de fechas) ──
        clave_disco = "ff5mom_all"
        if _cache_valida(clave_disco):
            logger.info("[FrenchDataFetcher] Cargando factores desde cache en disco...")
            resultado_completo = pd.read_parquet(_cache_path(clave_disco))
        else:
            # Descargar 5 factores (MKT, SMB, HML, RMW, CMA, RF)
            logger.info("[FrenchDataFetcher] Descargando 5 factores desde Kenneth French...")
            csv_5f = _descargar_csv_desde_zip(URLS['ff5_daily'])
            df_5f = _parsear_csv_french(csv_5f)

            # Descargar Momentum
            logger.info("[FrenchDataFetcher] Descargando factor Momentum...")
            csv_mom = _descargar_csv_desde_zip(URLS['momentum_daily'])
            df_mom = _parsear_csv_french(csv_mom)

            # Renombrar columnas al esquema interno
            df_5f = df_5f.rename(columns={'Mkt-RF': 'MKT'})
            df_mom = df_mom.rename(columns={'Mom': 'MOM'})

            # Combinar 5 factores + momentum (inner join por fechas comunes)
            resultado_completo = df_5f.join(df_mom[['MOM']], how='inner')

            # Persistir en disco
            try:
                resultado_completo.to_parquet(_cache_path(clave_disco))
                logger.info(f"[FrenchDataFetcher] Cache guardado en {_cache_path(clave_disco)}")
            except Exception as exc:
                logger.warning(f"[FrenchDataFetcher] No se pudo guardar cache en disco: {exc}")

        # Filtrar por rango de fechas
        resultado = resultado_completo.copy()
        if start_date:
            resultado = resultado.loc[start_date:]
        if end_date:
            resultado = resultado.loc[:end_date]

        logger.info(
            f"[FrenchDataFetcher] {len(resultado)} observaciones "
            f"({resultado.index.min().date()} → {resultado.index.max().date()})"
        )

        self._cache[clave_sesion] = resultado
        return resultado

    def clear_cache(self):
        """Limpia caché en memoria y en disco."""
        self._cache.clear()
        for parquet in _CACHE_DIR.glob("*.parquet"):
            try:
                parquet.unlink()
            except Exception:
                pass
        logger.info("[FrenchDataFetcher] Cache en memoria y disco limpiados")
