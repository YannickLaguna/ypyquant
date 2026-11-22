"""
ypyquant — API pública del paquete.

Importaciones directas disponibles:
    from ypyquant import BacktestEngine, BacktestConfig, BacktestResult, PortfolioBacktestService
    from ypyquant import AssetSelectionService, PortfolioOptimizationService
    from ypyquant import PerformanceMetrics, PerformanceAnalyzer
    from ypyquant import PortfolioOptimizationModel
    from ypyquant import WalkForwardEngine, ConfigWalkForward
    from ypyquant import BlackLittermanQuantLib, View
    from ypyquant import CAPMModel, FamaFrenchModel
    from ypyquant import TasaLibreRiesgoDinamica
    from ypyquant import FrenchDataFetcher
    from ypyquant import obtener_preset, listar_presets
    from ypyquant import SignalCombiner, TechnicalSignalGenerator, BaseSignalGenerator, SignalFrame
"""

__version__ = "0.5.0"

# ---------------------------------------------------------------------------
# Portfolio — backtesting y métricas
# ---------------------------------------------------------------------------
from ypyquant.data.french_data_fetcher import FrenchDataFetcher

# ---------------------------------------------------------------------------
# Data — tasas RF y factores
# ---------------------------------------------------------------------------
from ypyquant.data.risk_free_rates import TasaLibreRiesgoDinamica

# ---------------------------------------------------------------------------
# Execution — ejecución de trading (Fase 3+4)
# ---------------------------------------------------------------------------
from ypyquant.execution import (
    Broker,
    BrokerSimulado,
    PoliticaEjecucion,
    ProtectionManager,
    StorageInterface,
    TradeAnalytics,
    planificar_rebalanceo,
)

# ---------------------------------------------------------------------------
# Factors — CAPM y Fama-French
# ---------------------------------------------------------------------------
from ypyquant.factors.factor_models import CAPMModel, FamaFrenchModel
from ypyquant.optimization.black_litterman import (
    BlackLittermanQuantLib,
    View,
)

# ---------------------------------------------------------------------------
# Optimization — modelos y walk-forward
# ---------------------------------------------------------------------------
from ypyquant.optimization.optimizador_portafolio import PortfolioOptimizationService
from ypyquant.optimization.portfolio_models import PortfolioOptimizationModel
from ypyquant.optimization.presets import listar_presets, obtener_preset
from ypyquant.optimization.walk_forward import (
    ConfigWalkForward,
    WalkForwardEngine,
)
from ypyquant.portfolio.asset_selection import AssetSelectionService
from ypyquant.portfolio.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BacktestTrade,
    PortfolioBacktestService,
)
from ypyquant.portfolio.performance_metrics import (
    PerformanceAnalyzer,
    PerformanceMetrics,
)

# ---------------------------------------------------------------------------
# Signals — generación de señales (Fase 2)
# ---------------------------------------------------------------------------
from ypyquant.signals import (
    BaseSignalGenerator,
    FactorSignalGenerator,
    MLSignalGenerator,
    RegimeSignalGenerator,
    SignalCombiner,
    SignalFrame,
    TechnicalSignalGenerator,
)

# ---------------------------------------------------------------------------
# __all__ — exportaciones explícitas
# ---------------------------------------------------------------------------
__all__ = [
    # Backtesting
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "BacktestTrade",
    "PortfolioBacktestService",
    "AssetSelectionService",
    # Métricas
    "PerformanceMetrics",
    "PerformanceAnalyzer",
    # Optimización
    "PortfolioOptimizationModel",
    "PortfolioOptimizationService",
    "WalkForwardEngine",
    "ConfigWalkForward",
    "BlackLittermanQuantLib",
    "View",
    "obtener_preset",
    "listar_presets",
    # Factores
    "CAPMModel",
    "FamaFrenchModel",
    # Datos
    "TasaLibreRiesgoDinamica",
    "FrenchDataFetcher",
    # Señales (Fase 2)
    "BaseSignalGenerator",
    "SignalFrame",
    "TechnicalSignalGenerator",
    "MLSignalGenerator",
    "RegimeSignalGenerator",
    "FactorSignalGenerator",
    "SignalCombiner",
    # Execution (Fase 3+4)
    "Broker",
    "BrokerSimulado",
    "PoliticaEjecucion",
    "planificar_rebalanceo",
    "ProtectionManager",
    "TradeAnalytics",
    "StorageInterface",
]
