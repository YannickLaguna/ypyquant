"""
Presets Post-Modern Portfolio Theory (PMPT)
===========================================

PMPT mejora la teoría de Markowitz al considerar solo el riesgo de pérdidas
(downside risk) en lugar de la volatilidad total.

Modelos incluidos:
- PMPT con Semi-Varianza
- PMPT con CVaR (Conditional Value at Risk)

Estos modelos son más realistas ya que los inversores temen las pérdidas,
no la volatilidad al alza.
"""

from ypyquant.optimization.portfolio_models import (
    HistoricalReturnsModel,
    MarkowitzOptimizer,
    PMPTMinCVaROptimizer,
    PortfolioOptimizationModel,
    SampleCovarianceModel,
    SemiCovarianceModel,
)
from ypyquant.optimization.presets import registrar_preset


@registrar_preset('pmpt_semivarianza')
def crear_pmpt_semivarianza(risk_free_rate: float = 0.02) -> PortfolioOptimizationModel:
    """
    PMPT (Post-Modern Portfolio Theory) - Minimiza Downside Risk.

    Pipeline:
    1. Retornos: Media histórica

    2. Riesgo: Semi-Covarianza (solo retornos negativos)
       - En lugar de volatilidad total, solo considera downside
       - Matriz de covarianza calculada solo con retornos < 0

    3. Optimización: Min Volatilidad (pero sobre downside risk)
       - Minimiza riesgo de pérdidas
       - Ignora volatilidad al alza (deseable)

    Ventajas:
    - Más realista que Markowitz (los inversores temen pérdidas, no ganancias)
    - Mejor para inversores adversos al riesgo
    - Protección contra drawdowns

    Casos de uso:
    - Inversores conservadores
    - Protección contra pérdidas
    - Portfolios de preservación de capital

    Args:
        risk_free_rate: Tasa libre de riesgo para cálculos (default: 0.02)
    """
    return PortfolioOptimizationModel(
        name="PMPT (Semi-Varianza)",
        returns_model=HistoricalReturnsModel(method="mean", frequency=252),
        risk_model=SemiCovarianceModel(target_return=0.0, frequency=252),
        optimizer=MarkowitzOptimizer(objective="min_volatility", risk_free_rate=risk_free_rate),
        description="Post-Modern Portfolio Theory: minimiza downside risk en lugar de volatilidad total"
    )


@registrar_preset('pmpt_cvar')
def crear_pmpt_cvar(risk_free_rate: float = 0.02) -> PortfolioOptimizationModel:
    """
    PMPT con CVaR (Conditional Value at Risk).

    Pipeline:
    1. Retornos: Media histórica

    2. Riesgo: Covarianza muestral (solo para las métricas del resultado; el
       optimizador no la usa en su objetivo)
       - CVaR mide la pérdida esperada en el 5% peor de los casos
       - Más enfocado en tail risk que semi-varianza

    3. Optimización: PMPTMinCVaROptimizer, minimiza el CVaR empírico al 95%
       sobre el histórico de retornos (`optimize` se lo pasa como retornos_historicos)

    Ventajas:
    - Protección contra eventos extremos (tail risk)
    - Métrica de riesgo regulatoria (Basel)
    - Mejor que VaR para gestión de riesgo

    Casos de uso:
    - Gestión de riesgo extremo
    - Portfolios institucionales
    - Cumplimiento regulatorio (Basel III)

    Nota:
    - Hasta v0.3.0 este preset usaba semicovarianza + min_volatility (no minimizaba el CVaR);
      desde v0.4.0 usa PMPTMinCVaROptimizer.

    Args:
        risk_free_rate: Tasa libre de riesgo para cálculos (default: 0.02)
    """
    return PortfolioOptimizationModel(
        name="PMPT (CVaR 95%)",
        returns_model=HistoricalReturnsModel(method="mean", frequency=252),
        risk_model=SampleCovarianceModel(frequency=252),
        optimizer=PMPTMinCVaROptimizer(nivel_confianza=0.95, risk_free_rate=risk_free_rate),
        description="Post-Modern Portfolio Theory con CVaR (Conditional Value at Risk) al 95%"
    )
