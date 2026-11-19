"""
Estimación robusta de µ y optimización max-Sharpe (lógica del TFM «Optimización robusta de portafolios»).

Responsabilidad: punto de entrada público del subpaquete. Agrupa funciones puras
(sin estado, sin BD) portadas de la rejilla v2 del TFM:
- sigma.py:           Σ Ledoit-Wolf.
- estimadores_mu.py:  µ muestral, Bayes-Stein, BL-prior, BL-momentum, Michaud (+ diagnósticos).
- max_sharpe_qp.py:   max-Sharpe long-only vía QP exacto con cascada de respaldo.
- universo.py:        prioridad aleatoria, universos anidados y elegibilidad point-in-time.
- w_mercado.py:       w_mkt por capitalización inyectada o proxy por volumen.
- rejilla.py:         motor walk-forward de la rejilla (N, T) con datos inyectables.

Se relaciona con: optimization/portfolio_models.py (clases Strategy BayesSteinReturnsModel,
BLPriorReturnsModel, BLMomentumReturnsModel, MaxSharpeQPOptimizer, MichaudOptimizer),
optimization/presets/tfm.py (presets) y docs/migracion-v0.4.0.md.
"""

from ypyquant.optimization.estimacion_robusta.estimadores_mu import (
    calcular_diagnostico_bl,
    calcular_omega_momentum,
    estimar_mu_bayes_stein,
    estimar_mu_bl_momentum,
    estimar_mu_bl_prior,
    estimar_mu_michaud,
    estimar_mu_muestral,
    indices_bootstrap,
)
from ypyquant.optimization.estimacion_robusta.max_sharpe_qp import (
    optimizar_max_sharpe,
    optimizar_min_varianza,
    optimizar_volatilidad_objetivo,
)
from ypyquant.optimization.estimacion_robusta.rejilla import (
    ESTRATEGIAS_REJILLA,
    CargadorDatosRejilla,
    ConfiguracionRejilla,
    DatosRejilla,
    ResultadosRejilla,
    calcular_retorno_neto,
    calcular_turnover,
    ejecutar_rejilla,
    ejecutar_replica,
    preparar_panel,
)
from ypyquant.optimization.estimacion_robusta.sigma import estimar_sigma_ledoit_wolf
from ypyquant.optimization.estimacion_robusta.universo import (
    construir_mascara_membresia,
    elegibles_en_fecha,
    generar_prioridad_universo,
    seed_para,
    seleccionar_universo_anidado,
)
from ypyquant.optimization.estimacion_robusta.w_mercado import (
    CapitalizacionEstatica,
    ProveedorCapitalizacion,
    calcular_w_mkt_volumen,
    pesos_mercado,
)

__all__ = [
    "ESTRATEGIAS_REJILLA",
    "CapitalizacionEstatica",
    "CargadorDatosRejilla",
    "ConfiguracionRejilla",
    "DatosRejilla",
    "ProveedorCapitalizacion",
    "ResultadosRejilla",
    "calcular_diagnostico_bl",
    "calcular_omega_momentum",
    "calcular_retorno_neto",
    "calcular_turnover",
    "calcular_w_mkt_volumen",
    "construir_mascara_membresia",
    "ejecutar_rejilla",
    "ejecutar_replica",
    "elegibles_en_fecha",
    "estimar_mu_bayes_stein",
    "estimar_mu_bl_momentum",
    "estimar_mu_bl_prior",
    "estimar_mu_michaud",
    "estimar_mu_muestral",
    "estimar_sigma_ledoit_wolf",
    "generar_prioridad_universo",
    "indices_bootstrap",
    "optimizar_max_sharpe",
    "optimizar_min_varianza",
    "optimizar_volatilidad_objetivo",
    "pesos_mercado",
    "preparar_panel",
    "seed_para",
    "seleccionar_universo_anidado",
]
