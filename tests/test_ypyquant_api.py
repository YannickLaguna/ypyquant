"""
Tests de la API pública de ypyquant (src/ypyquant/__init__.py).

Responsabilidad:
    Verificar que las clases y funciones core se importan directamente desde el paquete
    sin rutas internas, y que __version__ y __all__ están definidos.

Se relaciona con:
    - src/ypyquant/__init__.py (código bajo prueba) y pyproject.toml (versión del paquete).
"""

# ---------------------------------------------------------------------------
# Imports directos desde ypyquant
# ---------------------------------------------------------------------------

def test_import_backtest_engine():
    from ypyquant import BacktestEngine
    assert BacktestEngine is not None


def test_import_backtest_config():
    from ypyquant import BacktestConfig
    assert BacktestConfig is not None


def test_import_backtest_result():
    from ypyquant import BacktestResult
    assert BacktestResult is not None


def test_import_performance_metrics():
    from ypyquant import PerformanceAnalyzer, PerformanceMetrics
    assert PerformanceMetrics is not None
    assert PerformanceAnalyzer is not None


def test_import_portfolio_optimization():
    from ypyquant import PortfolioOptimizationModel
    assert PortfolioOptimizationModel is not None


def test_import_walk_forward():
    from ypyquant import ConfigWalkForward, WalkForwardEngine
    assert WalkForwardEngine is not None
    assert ConfigWalkForward is not None


def test_import_black_litterman():
    from ypyquant import BlackLittermanQuantLib, View
    assert BlackLittermanQuantLib is not None
    assert View is not None


def test_import_factor_models():
    from ypyquant import CAPMModel, FamaFrenchModel
    assert CAPMModel is not None
    assert FamaFrenchModel is not None


def test_import_risk_free():
    from ypyquant import TasaLibreRiesgoDinamica
    assert TasaLibreRiesgoDinamica is not None


def test_import_french_fetcher():
    from ypyquant import FrenchDataFetcher
    assert FrenchDataFetcher is not None


def test_import_presets():
    from ypyquant import listar_presets, obtener_preset
    assert callable(obtener_preset)
    assert callable(listar_presets)


def test_listar_presets_returns_dict():
    from ypyquant import listar_presets
    presets = listar_presets()
    assert isinstance(presets, dict), "listar_presets debe devolver dict"
    assert len(presets) > 0, "Debe haber al menos un preset registrado"


def test_version():
    import ypyquant
    assert hasattr(ypyquant, "__version__"), "__version__ debe estar definido"
    assert ypyquant.__version__ == "0.2.0"


def test_all_defined():
    import ypyquant
    assert hasattr(ypyquant, "__all__"), "__all__ debe estar definido"
    assert len(ypyquant.__all__) > 0


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
