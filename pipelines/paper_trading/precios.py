"""
precios.py — Descarga de precios históricos de cierre ajustado (yfinance).

Responsabilidad:
    Devolver un DataFrame fecha x ticker con los cierres, formato que esperan los
    generadores de señales de ypyquant. Necesita red (extra `trading`).

    También decide CUÁNTA historia descargar: `dias_naturales_para_sesiones` convierte las sesiones que
    declara una estrategia (`Estrategia.sesiones_necesarias`) en días naturales, `resolver_dias_historico`
    combina eso con el mínimo configurado (env `NAUTILUS_LOOKBACK_DAYS`, `--dias-historico` o la API) y
    `avisar_historia_insuficiente` deja un WARNING claro si lo descargado aun así no basta (la estrategia
    caería en su caso de reserva, p. ej. pesos iguales, sin que se note).

Se relaciona con:
    - alpaca_vivo.py y backtest.py: lo usan para alimentar el combinador de señales y el modo registro.
    - ypyquant.estrategias.base: `Estrategia.sesiones_necesarias` y `MARGEN_SESIONES`.
    - config.py: `ConfiguracionServicio.dias_historico` (mínimo configurado).
"""

import logging
import math
from datetime import datetime, timedelta

import pandas as pd

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


def descargar_precios(tickers: list[str], dias_historico: int = 252) -> pd.DataFrame:
    """Descarga cierres ajustados de los últimos `dias_historico` días (más margen por festivos).

    Devuelve un DataFrame vacío si falla la descarga.
    """
    import yfinance as yf

    fin = datetime.now()
    inicio = fin - timedelta(days=dias_historico + 10)
    try:
        crudo = yf.download(
            tickers, start=inicio.strftime("%Y-%m-%d"), end=fin.strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False,
        )
        if crudo.empty:
            return pd.DataFrame()
        if len(tickers) == 1:
            precios = crudo[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            precios = crudo["Close"] if "Close" in crudo.columns else crudo.xs("Close", axis=1, level=0)
        precios = precios.dropna(how="all")
        logger.info("Precios descargados: %d días x %d tickers", precios.shape[0], precios.shape[1])
        return precios
    except Exception as e:
        logger.error("Error descargando precios: %s", e)
        return pd.DataFrame()
