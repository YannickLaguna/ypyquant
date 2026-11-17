"""
precios.py — Descarga de precios históricos de cierre ajustado (yfinance).

Responsabilidad:
    Devolver un DataFrame fecha x ticker con los cierres, formato que esperan los
    generadores de señales de ypyquant. Necesita red (extra `trading`).

Se relaciona con:
    - alpaca_vivo.py y backtest.py: lo usan para alimentar el combinador de señales.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


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
