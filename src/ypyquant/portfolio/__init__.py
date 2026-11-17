"""
Paquete portfolio: backtest, selección de activos, métricas de performance, rebalanceo,
Monte Carlo y bonos. Desde v0.3.0 es la única fuente de esta lógica (GrowSave importa de aquí).
"""
from ypyquant.portfolio.asset_selection import (
    AssetSelectionService,
    calcular_matriz_correlacion,
    calcular_metricas_ticker,
    diversify_clustering,
    diversify_distance,
    diversify_greedy,
    diversify_mixed,
    screen_by_momentum,
    screen_by_sharpe,
    screen_by_volatility,
)
from ypyquant.portfolio.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BacktestTrade,
    PortfolioBacktestService,
    format_backtest_result,
    format_backtest_results,
)
from ypyquant.portfolio.bond_pricing import BondPricer
from ypyquant.portfolio.monte_carlo import MonteCarloRiesgo
from ypyquant.portfolio.performance_metrics import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    calculate_var_quantlib,
    format_performance_metrics,
)

__all__ = [
    "AssetSelectionService",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BacktestTrade",
    "BondPricer",
    "MonteCarloRiesgo",
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "PortfolioBacktestService",
    "calcular_matriz_correlacion",
    "calcular_metricas_ticker",
    "calculate_var_quantlib",
    "diversify_clustering",
    "diversify_distance",
    "diversify_greedy",
    "diversify_mixed",
    "format_backtest_result",
    "format_backtest_results",
    "format_performance_metrics",
    "screen_by_momentum",
    "screen_by_sharpe",
    "screen_by_volatility",
]
