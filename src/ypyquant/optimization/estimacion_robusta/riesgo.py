"""
Primitivas de riesgo y paridad de riesgo compartidas por todo `ypyquant.optimization`.

Responsabilidad: ser la ÚNICA implementación de dos cálculos que antes existían duplicados:
- `var_cvar_historicos`: VaR y CVaR (Expected Shortfall) históricos diarios de una cartera. Antes los
  calculaban por separado `optimizador_portafolio.calcular_cvar` y
  `portfolio_models.PMPTMinCVaROptimizer._calcular_cvar` (mismo cálculo, resultados idénticos).
- `optimizar_paridad_riesgo`: contribuciones iguales al riesgo (mínimos cuadrados, SLSQP). Antes estaba
  en `optimizador_portafolio.optimizar_risk_parity` y en `portfolio_models.RiskParityOptimizer` con el
  mismo objetivo (diferían solo en `maxiter` y en la guarda de volatilidad nula).

Son funciones puras (numpy/pandas/scipy), sin estado ni dependencias del resto de `optimization`, para que
tanto el módulo de funciones (`optimizador_portafolio`) como las clases Strategy (`portfolio_models`) puedan
importarlas sin ciclos.

Lo que NO unifica (decisión pendiente, medido en tests/test_optimizacion_caracterizacion.py): max-Sharpe,
mínima varianza y volatilidad objetivo de `optimizador_portafolio` (SLSQP) frente a `max_sharpe_qp` (QP
exacto) difieren hasta 1e-4..6e-4 en pesos; cambiarlos alteraría los resultados que ve GrowSave.

Se relaciona con: optimizador_portafolio.py (calcular_cvar, optimizar_risk_parity),
portfolio_models.py (PMPTMinCVaROptimizer, RiskParityOptimizer) y
docs/planes/2026-11-21-optimizacion-canonica.md.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def var_cvar_historicos(
    retornos: pd.DataFrame, pesos: np.ndarray, confianza: float = 0.95, objetivo_diario: float = 0.0
) -> tuple[float, float]:
    """
    VaR y CVaR históricos DIARIOS de una cartera de pesos fijos.

    I: retornos diarios (T x N), pesos (N,), nivel de confianza y, opcional, un umbral diario que se resta
       a los retornos de la cartera (shortfall respecto a un objetivo).
    T: retorno de cartera → percentil (1-confianza) = VaR → media de la cola <= VaR = CVaR
       (si la cola queda vacía, CVaR = VaR).
    O: (var_diario, cvar_diario), ambos en unidades de retorno diario (negativos = pérdida).
    """
    retornos_cartera = (retornos * pesos).sum(axis=1)
    if objetivo_diario:
        retornos_cartera = retornos_cartera - objetivo_diario
    var = np.percentile(retornos_cartera, (1 - confianza) * 100)
    cola = retornos_cartera[retornos_cartera <= var]
    cvar = var if len(cola) == 0 else cola.mean()
    return var, cvar


def optimizar_paridad_riesgo(
    cov: pd.DataFrame | np.ndarray, peso_maximo: float = 1.0, maxiter: int | None = None
) -> tuple[np.ndarray, "object"]:
    """
    Paridad de riesgo: pesos long-only (suma 1, tope `peso_maximo`) con contribuciones iguales al riesgo.

    I: covarianza (N x N) anualizada, tope de peso y `maxiter` de SLSQP (None = el de scipy; el optimizador
       Strategy de `portfolio_models` usa 1000).
    T: minimiza Σ (c_i - σ/N)² con c_i = w_i·(Σw)_i/σ, desde la cartera equiponderada.
    O: (pesos, resultado_scipy); el resultado trae `success` y `message` para los metadatos.
    """
    matriz = np.asarray(cov, dtype=float)
    n = len(matriz)

    def objetivo(pesos):
        vol = np.sqrt(np.dot(pesos.T, np.dot(matriz, pesos)))
        if vol == 0:
            return 1e10
        contribucion = pesos * (np.dot(matriz, pesos) / vol)
        return np.sum((contribucion - vol / n) ** 2)

    opciones = {'maxiter': maxiter} if maxiter is not None else {}
    resultado = minimize(
        objetivo, np.array([1 / n] * n), method='SLSQP', bounds=tuple((0, peso_maximo) for _ in range(n)),
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}], options=opciones,
    )
    return resultado.x, resultado
