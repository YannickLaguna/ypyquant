"""
Presets Tradicionales de Optimización de Portafolios
====================================================

Modelos clásicos basados en teoría moderna de portafolios (MPT):
- Markowitz Clásico
- Risk Parity Robusto

Estos presets usan técnicas establecidas y probadas por décadas.
"""

from ypyquant.optimization.portfolio_models import (
    PortfolioOptimizationModel,
    HistoricalReturnsModel,
    LedoitWolfShrinkageModel,
    MarkowitzOptimizer,
    RiskParityOptimizer
)
from ypyquant.optimization.presets import registrar_preset


@registrar_preset('markowitz')
def crear_markowitz_clasico(risk_free_rate: float = 0.02) -> PortfolioOptimizationModel:
    """
    Modelo clásico de Markowitz (1952) con Ledoit-Wolf Shrinkage.

    Pipeline:
    - Retornos: Media histórica
    - Riesgo: Ledoit-Wolf shrinkage (matriz estable y robusta)
    - Optimización: Maximizar Sharpe Ratio

    Características:
    - Estable incluso con muchos activos
    - Matriz de covarianza bien condicionada
    - Óptimo teóricamente demostrado (Ledoit & Wolf 2004)

    Casos de uso:
    - Portfolios de cualquier tamaño
    - Datos ruidosos o con outliers
    - Necesidad de estabilidad numérica

    Args:
        risk_free_rate: Tasa libre de riesgo para cálculo de Sharpe (default: 0.02)
    """
    return PortfolioOptimizationModel(
        name="Markowitz con Ledoit-Wolf",
        returns_model=HistoricalReturnsModel(method="mean", frequency=252),
        risk_model=LedoitWolfShrinkageModel(frequency=252),
        optimizer=MarkowitzOptimizer(objective="max_sharpe", risk_free_rate=risk_free_rate),
        description="Modelo Markowitz con shrinkage óptimo de Ledoit-Wolf"
    )


@registrar_preset('risk_parity')
def crear_risk_parity_robusto(risk_free_rate: float = 0.02) -> PortfolioOptimizationModel:
    """
    Modelo robusto con Risk Parity.

    Pipeline:
    - Retornos: Mediana histórica (más robusta a outliers)
    - Riesgo: Ledoit-Wolf shrinkage (matriz estable)
    - Optimización: Risk Parity (cada activo contribuye igual al riesgo)

    Características:
    - Robusto a outliers
    - Buena diversificación
    - No depende de predicciones de retorno

    Casos de uso:
    - Portfolios grandes (> 30 activos)
    - Datos ruidosos
    - Incertidumbre alta sobre retornos futuros
    """
    return PortfolioOptimizationModel(
        name="Risk Parity Robusto",
        returns_model=HistoricalReturnsModel(method="median", frequency=252),
        risk_model=LedoitWolfShrinkageModel(frequency=252),
        optimizer=RiskParityOptimizer(),
        description="Modelo robusto con shrinkage y paridad de riesgo"
    )
