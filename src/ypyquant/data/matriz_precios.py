"""
Matriz de precios y retornos a partir de un cargador de precios inyectado.

Responsabilidad:
- `CargadorPrecios`: contrato `(ticker, fecha_inicio, fecha_fin) -> DataFrame OHLCV | None`.
  ypyquant no accede a ninguna BD: la aplicación inyecta su función (en GrowSave,
  `app.DB.database.get_ticker_data`, que devuelve columnas en minúscula).
- `extraer_serie_precios`: elige el cierre ajustado (adjusted_close > Adj Close > close > Close).
- `con_columna_close`: añade la columna 'Close' que usa el motor de backtest (arreglo de
  GrowSave del 2026-11-15: sin ella ningún día tenía precio y había 0 rebalanceos).
- `construir_matriz_precios` y `calcular_matriz_retornos`: lo que antes duplicaban
  AssetSelectionService y PortfolioOptimizationService de GrowSave (_get_prices_matrix,
  _get_returns_matrix), con el preprocesador centralizado ProcesadorDatos.

Se relaciona con:
- ypyquant.optimization.data_preprocessing (ProcesadorDatos).
- ypyquant.portfolio.asset_selection, ypyquant.portfolio.backtest_engine,
  ypyquant.optimization.optimizador_portafolio (consumidores).
"""

import logging
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from ypyquant.optimization.data_preprocessing import ProcesadorDatos

logger = logging.getLogger(__name__)

# (ticker, fecha_inicio 'YYYY-MM-DD', fecha_fin 'YYYY-MM-DD') -> DataFrame OHLCV indexado por fecha
CargadorPrecios = Callable[[str, str, str], pd.DataFrame | None]

# Orden de preferencia de columnas de precio: siempre el cierre ajustado (dividendos y splits)
COLUMNAS_PRECIO = ('adjusted_close', 'Adj Close', 'close', 'Close')


def extraer_serie_precios(ohlcv: pd.DataFrame | None, ticker: str = '', avisar: bool = True) -> pd.Series | None:
    """
    Devuelve la serie de precios de un OHLCV priorizando el cierre ajustado.

    Args:
        avisar: si True, registra un warning cuando no hay cierre ajustado (el screening,
            que recorre cientos de tickers, lo desactiva).

    Returns:
        Serie de precios o None si el DataFrame está vacío o no tiene columna de precio.
    """
    if ohlcv is None or ohlcv.empty:
        return None
    for columna in COLUMNAS_PRECIO:
        if columna in ohlcv.columns:
            if avisar and columna in ('close', 'Close'):
                logger.warning(f"{ticker}: adjusted_close no disponible, usando {columna}")
            return ohlcv[columna]
    logger.error(f"No se encontró columna de precios para {ticker}")
    return None


def con_columna_close(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Garantiza la columna 'Close' que usa el motor de backtest.

    Si no existe, se copia del cierre ajustado ('adjusted_close' o 'Adj Close') o, en su
    defecto, de 'close'. Si ya existe o el DataFrame está vacío, se devuelve tal cual.
    """
    if df is None or df.empty or 'Close' in df.columns:
        return df
    df = df.copy()
    for columna in ('adjusted_close', 'Adj Close', 'close'):
        if columna in df.columns:
            df['Close'] = df[columna]
            break
    return df


def construir_matriz_precios(
    tickers: list[str],
    fecha_inicio: str,
    fecha_fin: str,
    cargador_precios: CargadorPrecios,
    estricto: bool = False,
) -> pd.DataFrame:
    """
    Construye la matriz de precios (columnas = tickers, índice = fechas) con filas completas.

    Args:
        tickers: símbolos a cargar (se respeta su orden en las columnas).
        fecha_inicio / fecha_fin: rango 'YYYY-MM-DD'.
        cargador_precios: función de carga inyectada.
        estricto: si True y ningún ticker tiene datos, lanza ValueError (comportamiento del
            servicio de optimización); si False devuelve un DataFrame vacío (selección de activos).

    Returns:
        DataFrame de precios sin NaN (se eliminan las fechas en que falte algún ticker).
    """
    precios = {}
    for ticker in tickers:
        try:
            serie = extraer_serie_precios(cargador_precios(ticker, fecha_inicio, fecha_fin), ticker)
        except Exception as exc:  # noqa: BLE001 — un ticker defectuoso no debe abortar el lote
            logger.error(f"Error obteniendo precios para {ticker}: {exc}")
            continue
        if serie is None:
            logger.debug(f"No hay datos disponibles para {ticker}")
            continue
        precios[ticker] = serie

    if not precios:
        if estricto:
            raise ValueError("No se encontraron datos válidos para ningún ticker")
        return pd.DataFrame()

    return pd.DataFrame(precios).dropna()


def calcular_matriz_retornos(
    precios_df: pd.DataFrame,
    tipo_retorno: Literal["log", "simple"] = "log",
    ratio_minimo_validos: float = 0.95,
    estricto: bool = True,
) -> pd.DataFrame:
    """
    Convierte una matriz de precios en retornos diarios limpios con ProcesadorDatos.

    Args:
        precios_df: matriz de precios.
        tipo_retorno: 'log' (por defecto en optimización) o 'simple'.
        ratio_minimo_validos: fracción mínima de datos válidos por ticker (0.95 en
            optimización, 0.90 en screening).
        estricto: True = comportamiento de optimización (los errores del preprocesador se
            propagan); False = comportamiento de selección de activos (con < 2 tickers calcula
            los retornos a mano y, si el preprocesador falla, devuelve un DataFrame vacío).

    Returns:
        DataFrame de retornos diarios (columnas = tickers que superan la validación).
    """
    if precios_df.empty:
        return pd.DataFrame()

    if not estricto and len(precios_df.columns) < 2:
        logger.warning("Menos de 2 tickers con datos válidos")
        if tipo_retorno == "log":
            retornos = np.log(precios_df / precios_df.shift(1)).iloc[1:]
        else:
            retornos = precios_df.pct_change().iloc[1:]
        return retornos.dropna()

    procesador = ProcesadorDatos(
        tipo_retorno=tipo_retorno,
        metodo_llenar_na="ffill",
        remover_outliers=True,
        umbral_std_outliers=5.0,
        ratio_minimo_validos=ratio_minimo_validos,
    )
    try:
        retornos, metadatos = procesador.procesar(precios_df, validar=True)
    except ValueError as exc:
        if estricto:
            raise
        logger.warning(f"Error en preprocesamiento: {exc}. Retornando matriz vacía.")
        return pd.DataFrame()

    if metadatos.get('tickers_removidos'):
        logger.warning(f"Tickers removidos por baja calidad: {metadatos['tickers_removidos']}")
    for advertencia in metadatos.get('advertencias', []) or []:
        logger.warning(advertencia)
    return retornos
