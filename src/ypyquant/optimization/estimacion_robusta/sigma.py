"""
Estimación de la matriz de covarianza Σ para el subpaquete `estimacion_robusta`.

Responsabilidad: ofrecer `estimar_sigma_ledoit_wolf`, el shrinkage Ledoit-Wolf
(target identidad escalada, intensidad analítica) que el TFM usa como Σ común
de todas las estrategias, parametrizado por la frecuencia de los datos
(12 = mensual, la del TFM; 252 = diario, la del resto de ypyquant).

Equivalencia con `LedoitWolfShrinkageModel` (optimization/portfolio_models.py):
ambos llaman a `pypfopt.risk_models.CovarianceShrinkage(returns_data=True,
frequency=f).ledoit_wolf()`, así que para los mismos datos y la misma
frecuencia el resultado es idéntico bit a bit (lo comprueba
tests/test_estimacion_robusta.py). Las únicas diferencias son de contrato:
el modelo de portfolio_models exige un DataFrame sin NaN (lanza ValueError) y
tiene como reserva sklearn; esta función acepta NaN aislados (pypfopt los
tolera según su versión) y cae a la covarianza muestral si pypfopt falla,
exactamente como el original del TFM.

Se relaciona con: estimadores_mu.py (Bayes-Stein, BL y Michaud necesitan Σ),
max_sharpe_qp.py (consume Σ), rejilla.py (la estima en cada celda/fecha) y
portfolio_models.LedoitWolfShrinkageModel (equivalente numérico).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

FRECUENCIA_MENSUAL = 12
FRECUENCIA_DIARIA = 252


def estimar_sigma_ledoit_wolf(retornos: pd.DataFrame, frecuencia: int = FRECUENCIA_MENSUAL) -> pd.DataFrame:
    """
    Covarianza con shrinkage Ledoit-Wolf, anualizada con `frecuencia`.

    Parámetros
    ----------
    retornos : DataFrame (T x N) de retornos (columnas = tickers) en la frecuencia indicada.
    frecuencia : observaciones por año (12 mensual, 252 diaria).

    Retorna
    -------
    DataFrame (N x N) con la covarianza anualizada. Si pypfopt falla, devuelve la
    covarianza muestral anualizada y deja un aviso en el log (comportamiento del TFM).
    """
    try:
        from pypfopt import risk_models

        return risk_models.CovarianceShrinkage(retornos, returns_data=True, frequency=frecuencia).ledoit_wolf()
    except Exception as exc:  # noqa: BLE001 - reserva deliberada, igual que el original
        logger.warning("[LedoitWolf] pypfopt falló (%s), usando covarianza muestral.", exc)
        return retornos.cov() * frecuencia
