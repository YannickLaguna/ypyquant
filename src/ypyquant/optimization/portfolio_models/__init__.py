"""
Modelos de retorno, riesgo y optimización de portafolios (Strategy Pattern), repartidos por tipo.

Responsabilidad: reexportar con la misma ruta de siempre (`ypyquant.optimization.portfolio_models`) todo lo que
antes vivía en el módulo único de ~1.900 líneas, de modo que ninguna importación existente (GrowSave incluido)
se rompa. La implementación está en:
- base.py:            enums (`ReturnModelType`, `RiskModelType`, `OptimizerType`) y clases abstractas.
- modelos_retorno.py: estimadores de µ (históricos, CAPM, Black-Litterman, LSTM, Bayes-Stein, BL-prior/momentum).
- modelos_riesgo.py:  estimadores de Σ (muestral, Ledoit-Wolf, exponencial, semicovarianzas, PCA).
- optimizadores.py:   Markowitz, paridad de riesgo, PMPT, max-Sharpe QP, Michaud, robustos.
- modelo.py:          `PortfolioOptimizationModel` (µ + Σ + optimizador) y fábricas `create_*`.

Se relaciona con: optimizador_portafolio.py (`crear_modelo_personalizado`, servicio), presets/, walk_forward.py,
estimacion_robusta/ (lógica canónica de los algoritmos que envuelven) y tests/test_optimizacion_compatibilidad.py
(comprueba que todos los nombres antiguos siguen importables).
"""

from ypyquant.optimization.portfolio_models.base import (
    ExpectedReturnsModel,
    OptimizerType,
    PortfolioOptimizer,
    ReturnModelType,
    RiskModel,
    RiskModelType,
)
from ypyquant.optimization.portfolio_models.modelo import (
    PortfolioOptimizationModel,
    create_capm_model,
    create_classic_markowitz_model,
    create_robust_model,
)
from ypyquant.optimization.portfolio_models.modelos_retorno import (
    BayesSteinReturnsModel,
    BlackLittermanReturnsModel,
    BLMomentumReturnsModel,
    BLPriorReturnsModel,
    CAPMReturnsModel,
    HistoricalReturnsModel,
    LSTMReturnsModel,
)
from ypyquant.optimization.portfolio_models.modelos_riesgo import (
    ExponentialCovarianceModel,
    LedoitWolfSemiCovarianceModel,
    LedoitWolfShrinkageModel,
    PCADenoisedRiskModel,
    SampleCovarianceModel,
    SemiCovarianceModel,
)
from ypyquant.optimization.portfolio_models.optimizadores import (
    MarkowitzOptimizer,
    MaxSharpeQPOptimizer,
    MichaudOptimizer,
    PMPTMaxSortinoOptimizer,
    PMPTMinCVaROptimizer,
    RiskParityOptimizer,
    RobustoCajaOptimizer,
    RobustoElipsoideOptimizer,
)

__all__ = [
    "BLMomentumReturnsModel",
    "BLPriorReturnsModel",
    "BayesSteinReturnsModel",
    "BlackLittermanReturnsModel",
    "CAPMReturnsModel",
    "ExpectedReturnsModel",
    "ExponentialCovarianceModel",
    "HistoricalReturnsModel",
    "LSTMReturnsModel",
    "LedoitWolfSemiCovarianceModel",
    "LedoitWolfShrinkageModel",
    "MarkowitzOptimizer",
    "MaxSharpeQPOptimizer",
    "MichaudOptimizer",
    "OptimizerType",
    "PCADenoisedRiskModel",
    "PMPTMaxSortinoOptimizer",
    "PMPTMinCVaROptimizer",
    "PortfolioOptimizationModel",
    "PortfolioOptimizer",
    "ReturnModelType",
    "RiskModel",
    "RiskModelType",
    "RiskParityOptimizer",
    "RobustoCajaOptimizer",
    "RobustoElipsoideOptimizer",
    "SampleCovarianceModel",
    "SemiCovarianceModel",
    "create_capm_model",
    "create_classic_markowitz_model",
    "create_robust_model",
]
