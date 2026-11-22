"""
precios.py — Precios históricos de cierre ajustado para el bot, desde la MISMA fuente que el motor.

Responsabilidad:
    Devolver un DataFrame fecha x ticker con los cierres, formato que esperan los
    generadores de señales de ypyquant. Los datos salen de `FuentePrecios` (por defecto
    `fuente_por_defecto()`: yfinance con caché Parquet, la que usan los backtests de la API), así que el bot y
    el backtest ven los mismos precios; ya no hay un `yf.download` propio. Sin caché ni red devuelve vacío.

    También decide CUÁNTA historia descargar: `dias_naturales_para_sesiones` convierte las sesiones que
    declara una estrategia (`Estrategia.sesiones_necesarias`) en días naturales, `resolver_dias_historico`
    combina eso con el mínimo configurado (env `NAUTILUS_LOOKBACK_DAYS`, `--dias-historico` o la API) y
    `avisar_historia_insuficiente` deja un WARNING claro si lo descargado aun así no basta (la estrategia
    caería en su caso de reserva, p. ej. pesos iguales, sin que se note).

Se relaciona con:
    - alpaca_vivo.py y backtest.py: lo usan para alimentar el combinador de señales y el modo registro.
    - ypyquant.data.fuente_precios y ypyquant.data.fuentes (la fuente de precios compartida con la API).
    - ypyquant.estrategias.base: `Estrategia.sesiones_necesarias` y `MARGEN_SESIONES`.
    - config.py: `ConfiguracionServicio.dias_historico` (mínimo configurado).
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd

from ypyquant.data.fuente_precios import FuentePrecios
from ypyquant.data.fuentes import fuente_por_defecto

logger = logging.getLogger(__name__)

SESIONES_POR_ANIO = 252
DIAS_POR_ANIO = 365
# Días naturales extra sobre la conversión sesiones -> calendario (festivos largos, descarga con fin exclusivo)
MARGEN_DIAS_NATURALES = 15


def dias_naturales_para_sesiones(sesiones: int) -> int:
    """
    Días naturales que hay que pedir para obtener al menos `sesiones` sesiones de mercado.

    Un año tiene ~252 sesiones en 365 días naturales; se redondea hacia arriba y se suma
    `MARGEN_DIAS_NATURALES`. 0 sesiones (sin requisito) devuelve 0.
    """
    if sesiones <= 0:
        return 0
    return math.ceil(sesiones * DIAS_POR_ANIO / SESIONES_POR_ANIO) + MARGEN_DIAS_NATURALES


def resolver_dias_historico(sesiones_necesarias: int, dias_configurados: int) -> int:
    """
    Días naturales a descargar: el mayor entre lo que necesita la estrategia y el mínimo configurado.

    El valor configurado (env, CLI o API) actúa como MÍNIMO explícito: si pides más historia que la
    necesaria se respeta; si pides menos, se amplía hasta lo que la estrategia necesita (con menos
    daría un resultado distinto del que describe la estrategia). Una estrategia sin requisito (0
    sesiones) usa solo el configurado.
    """
    return max(dias_naturales_para_sesiones(sesiones_necesarias), dias_configurados)


def avisar_historia_insuficiente(precios: pd.DataFrame, sesiones_necesarias: int, nombre: str) -> bool:
    """
    WARNING claro si lo descargado no alcanza las sesiones que necesita la estrategia; True si falta.

    Comprueba el total de sesiones y, por separado, los tickers con menos historia (p. ej. salidas a bolsa
    recientes). Cuando falta, la estrategia usa su caso de reserva (momentum y los estimadores del TFM
    reparten a pesos iguales o con una ventana más corta): el bot sigue operando, pero ya no es la
    estrategia que se pidió.
    """
    if sesiones_necesarias <= 0 or precios.empty:
        return False
    total = len(precios)
    cortos = {t: int(precios[t].notna().sum()) for t in precios.columns if precios[t].notna().sum() < sesiones_necesarias}
    if total >= sesiones_necesarias and not cortos:
        return False
    if total < sesiones_necesarias:
        detalle = f"se descargaron {total} sesiones y necesita {sesiones_necesarias}"
    else:
        detalle = f"hay {total} sesiones, pero a estos tickers les faltan (sesiones propias: {cortos}) y necesita {sesiones_necesarias}"
    logger.warning(
        "Historia insuficiente para %s: %s. La estrategia cae en su caso de reserva (p. ej. pesos iguales o una "
        "ventana más corta) y los pesos NO son los que describe. Aumenta los días de historia o revisa los tickers.",
        nombre, detalle,
    )
    return True


def descargar_precios(tickers: list[str], dias_historico: int = 252, fuente: FuentePrecios | None = None) -> pd.DataFrame:
    """Cierres ajustados de los últimos `dias_historico` días (más margen por festivos) hasta AYER, de la fuente de precios.

    `fuente` por defecto es `fuente_por_defecto()` (la misma que el motor de backtests). El día en curso queda
    fuera: su barra está incompleta mientras el mercado está abierto. Devuelve un DataFrame vacío si no hay datos
    de ningún ticker; los tickers sin datos se omiten con un aviso.
    """
    fuente = fuente or fuente_por_defecto()
    hoy = datetime.now()
    inicio = (hoy - timedelta(days=dias_historico + 10)).strftime("%Y-%m-%d")
    fin = (hoy - timedelta(days=1)).strftime("%Y-%m-%d")

    def cargar(ticker: str) -> pd.Series | None:
        try:
            df = fuente.ohlcv(ticker, inicio, fin, "ajustado")
        except Exception as e:  # noqa: BLE001 - un ticker roto no debe tumbar la descarga de los demás
            logger.error("Error obteniendo precios de %s: %s", ticker, e)
            return None
        return None if df is None or df.empty else df["adjusted_close"].rename(ticker)

    with ThreadPoolExecutor(max_workers=8) as pool:
        series = list(pool.map(cargar, tickers))
    obtenidas = [serie for serie in series if serie is not None]
    for ticker, serie in zip(tickers, series, strict=True):
        if serie is None:
            logger.warning("Sin precios de %s; se omite", ticker)
    if not obtenidas:
        return pd.DataFrame()
    precios = pd.concat(obtenidas, axis=1).dropna(how="all")
    logger.info("Precios cargados: %d días x %d tickers", precios.shape[0], precios.shape[1])
    return precios
