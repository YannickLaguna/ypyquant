"""
Presets del TFM «Optimización robusta de portafolios» (v0.4.0).

Responsabilidad: registrar tres modelos listos para usar basados en la estimación robusta
de µ, con Σ Ledoit-Wolf y optimizador max-Sharpe exacto (QP) o remuestreo de Michaud:
- `tfm_bayes_stein`: µ de Bayes-Stein (Jorion) + max-Sharpe QP.
- `tfm_michaud`:     remuestreo eficiente (bootstrap iid, 50 réplicas) con pesos promediados.
- `tfm_bl_prior`:    Π = δ·Σ·w_mkt + max-Sharpe QP; sin views equivale a la cartera de mercado.
                      Necesita `market_caps` (kwarg de `optimize`); si no llegan usa el proxy por
                      volumen (si se pasan `precios` y `volumen`) o equiponderado, y avisa en
                      `optimizer_metadata['avisos']`.

Convención: los presets usan datos diarios anualizados con 252 (como el resto de ypyquant).
Para los datos mensuales del TFM construye los modelos con `frequency=12` (ver
docs/migracion-v0.4.0.md) o usa el motor `estimacion_robusta.rejilla`.
`tfm_bl_prior` fija rf=0 en el optimizador porque Π ya es un exceso de retorno (con rf>0 los
pesos dejarían de ser w_mkt); aplicar (1+τ) a Σ no cambia los pesos y no se aplica.

Se relaciona con: presets/__init__.py (registro), portfolio_models.py (clases Strategy) y
estimacion_robusta/ (lógica).
"""

from ypyquant.optimization.portfolio_models import (
    BayesSteinReturnsModel,
    BLPriorReturnsModel,
    HistoricalReturnsModel,
    LedoitWolfShrinkageModel,
    MaxSharpeQPOptimizer,
    MichaudOptimizer,
    PortfolioOptimizationModel,
)
from ypyquant.optimization.presets import registrar_preset


@registrar_preset('tfm_bayes_stein')
def crear_tfm_bayes_stein(risk_free_rate: float = 0.02) -> PortfolioOptimizationModel:
    """µ de Bayes-Stein + Σ Ledoit-Wolf + max-Sharpe QP (reduce el error de estimación de µ̂)."""
    return PortfolioOptimizationModel(
        name="TFM Bayes-Stein",
        returns_model=BayesSteinReturnsModel(frequency=252),
        risk_model=LedoitWolfShrinkageModel(frequency=252),
        optimizer=MaxSharpeQPOptimizer(risk_free_rate=risk_free_rate),
        description="Bayes-Stein (Jorion 1986) sobre µ, shrinkage Ledoit-Wolf en Σ y max-Sharpe long-only exacto (QP)",
    )


@registrar_preset('tfm_michaud')
def crear_tfm_michaud(risk_free_rate: float = 0.02) -> PortfolioOptimizationModel:
    """Remuestreo eficiente de Michaud: bootstrap del histórico, refit µ/Σ_LW y promedio de pesos."""
    return PortfolioOptimizationModel(
        name="TFM Michaud",
        returns_model=HistoricalReturnsModel(method="mean", frequency=252),
        risk_model=LedoitWolfShrinkageModel(frequency=252),
        optimizer=MichaudOptimizer(n_replicas=50, metodo_bootstrap="iid", frequency=252,
                                   risk_free_rate=risk_free_rate),
        description="Remuestreo eficiente de Michaud (bootstrap iid, 50 réplicas, Σ Ledoit-Wolf por réplica)",
    )


@registrar_preset('tfm_bl_prior')
def crear_tfm_bl_prior(risk_free_rate: float = 0.02) -> PortfolioOptimizationModel:
    """BL-prior sin views: Π = δΣw_mkt; el max-Sharpe devuelve la cartera de mercado (`risk_free_rate` no se usa)."""
    return PortfolioOptimizationModel(
        name="TFM BL-prior",
        returns_model=BLPriorReturnsModel(risk_aversion=2.5, frequency=252),
        risk_model=LedoitWolfShrinkageModel(frequency=252),
        optimizer=MaxSharpeQPOptimizer(risk_free_rate=0.0),
        description="Black-Litterman prior de equilibrio (δ=2.5) sin views: reproduce w_mkt; requiere market_caps",
    )
