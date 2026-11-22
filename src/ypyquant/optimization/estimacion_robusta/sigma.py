"""
Estimación de la matriz de covarianza Σ para el subpaquete `estimacion_robusta`.

Responsabilidad: ofrecer `estimar_sigma_ledoit_wolf`, el shrinkage Ledoit-Wolf
(target identidad escalada, intensidad analítica) que el TFM usa como Σ común
de todas las estrategias, parametrizado por la frecuencia de los datos
(12 = mensual, la del TFM; 252 = diario, la del resto de ypyquant).

Una sola implementación: `LedoitWolfShrinkageModel` (optimization/portfolio_models/modelos_riesgo.py)
delega en `ledoit_wolf_con_metodo`, así que función y modelo dan el mismo resultado y tienen la MISMA
reserva ante fallo: si pypfopt no está o no puede estimar (ImportError, ValueError, ArithmeticError,
LinAlgError), se devuelve la covarianza muestral anualizada con un aviso en el log y se anota el
método usado. Las únicas diferencias de contrato: el modelo exige un DataFrame sin NaN (lanza
ValueError) y la función acepta NaN aislados (pypfopt los tolera según su versión). Unificado el
2026-11-22 (antes el modelo tenía como reserva sklearn).

Se relaciona con: estimadores_mu.py (Bayes-Stein, BL y Michaud necesitan Σ),
max_sharpe_qp.py (consume Σ), rejilla.py (la estima en cada celda/fecha) y
portfolio_models.LedoitWolfShrinkageModel (delega en `ledoit_wolf_con_metodo`).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FRECUENCIA_MENSUAL = 12
FRECUENCIA_DIARIA = 252


# Fallos de pypfopt que justifican la reserva muestral (no se capturan errores de programación como
# TypeError o AttributeError): pypfopt ausente, datos degenerados, aritmética o álgebra lineal.
_ERRORES_ESTIMACION = (ImportError, ValueError, ArithmeticError, np.linalg.LinAlgError)


def ledoit_wolf_con_metodo(retornos: pd.DataFrame, frecuencia: int = FRECUENCIA_MENSUAL) -> tuple[pd.DataFrame, str]:
    """
    Núcleo del shrinkage Ledoit-Wolf: devuelve `(covarianza anualizada, método)`, con método
    `"pypfopt"` o `"fallback"` (covarianza muestral anualizada si pypfopt falla; deja un aviso en el log).
    """
    try:
        from pypfopt import risk_models

        cov = risk_models.CovarianceShrinkage(retornos, returns_data=True, frequency=frecuencia).ledoit_wolf()
        return cov, "pypfopt"
    except _ERRORES_ESTIMACION as exc:  # reserva deliberada y acotada: ver _ERRORES_ESTIMACION
        logger.warning("[LedoitWolf] pypfopt falló (%s), usando covarianza muestral.", exc)
        return retornos.cov() * frecuencia, "fallback"


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
    return ledoit_wolf_con_metodo(retornos, frecuencia)[0]
