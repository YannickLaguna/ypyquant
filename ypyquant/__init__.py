"""
ypyquant — API pública del paquete.

Importaciones directas disponibles:
    from ypyquant import BacktestEngine, BacktestConfig, BacktestResult
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

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Portfolio — backtesting y métricas
# ---------------------------------------------------------------------------
from ypyquant.portfolio.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
)
from ypyquant.portfolio.performance_metrics import (
    PerformanceMetrics,
    PerformanceAnalyzer,
)

# ---------------------------------------------------------------------------
# Optimization — modelos y walk-forward
# ---------------------------------------------------------------------------
from ypyquant.optimization.portfolio_models import PortfolioOptimizationModel
from ypyquant.optimization.walk_forward import (
    WalkForwardEngine,
    ConfigWalkForward,
)
from ypyquant.optimization.black_litterman import (
    BlackLittermanQuantLib,
    View,
)
from ypyquant.optimization.presets import obtener_preset, listar_presets

# ---------------------------------------------------------------------------
# Factors — CAPM y Fama-French
# ---------------------------------------------------------------------------
from ypyquant.factors.factor_models import CAPMModel, FamaFrenchModel

# ---------------------------------------------------------------------------
# Data — tasas RF y factores
# ---------------------------------------------------------------------------
from ypyquant.data.risk_free_rates import TasaLibreRiesgoDinamica
from ypyquant.data.french_data_fetcher import FrenchDataFetcher

# ---------------------------------------------------------------------------
# Signals — generación de señales (Fase 2)
# ---------------------------------------------------------------------------
from ypyquant.signals import (
    BaseSignalGenerator,
    SignalFrame,
    TechnicalSignalGenerator,
    MLSignalGenerator,
    RegimeSignalGenerator,
    FactorSignalGenerator,
    SignalCombiner,
)

# ---------------------------------------------------------------------------
# Execution — ejecución de trading (Fase 3+4)
# ---------------------------------------------------------------------------
from ypyquant.execution import (
    BrokerInterface,
    PaperBroker,
    BotExecutor,
    ProtectionManager,
    WalletManager,
    TradeAnalytics,
    StorageInterface,
    OrderSide,
    OrderType,
    OrderStatus,
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
    # Métricas
    "PerformanceMetrics",
    "PerformanceAnalyzer",
    # Optimización
    "PortfolioOptimizationModel",
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
    "BrokerInterface",
    "PaperBroker",
    "BotExecutor",
    "ProtectionManager",
    "WalletManager",
    "TradeAnalytics",
    "StorageInterface",
    "OrderSide",
    "OrderType",
    "OrderStatus",
]
