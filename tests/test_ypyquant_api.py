"""
Tests para ypyquant/__init__.py — API pública del paquete.

Verifica que todas las clases core son importables directamente
desde el paquete sin rutas internas.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "PASS"
FAIL = "FAIL"
results = []


def test(name: str, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  [PASS] {name}")
    except Exception as exc:
        results.append((FAIL, name))
        print(f"  [FAIL] {name}")
        traceback.print_exc()


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
    from ypyquant import PerformanceMetrics, PerformanceAnalyzer
    assert PerformanceMetrics is not None
    assert PerformanceAnalyzer is not None


def test_import_portfolio_optimization():
    from ypyquant import PortfolioOptimizationModel
    assert PortfolioOptimizationModel is not None


def test_import_walk_forward():
    from ypyquant import WalkForwardEngine, ConfigWalkForward
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
    from ypyquant import obtener_preset, listar_presets
    assert callable(obtener_preset)
    assert callable(listar_presets)


def test_listar_presets_returns_dict():
    from ypyquant import listar_presets
    presets = listar_presets()
    assert isinstance(presets, dict), "listar_presets debe devolver dict"
    assert len(presets) > 0, "Debe haber al menos un preset registrado"
    print(f"    Presets disponibles: {list(presets.keys())}")


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

if __name__ == "__main__":
    print("\n=== test_ypyquant_api.py ===\n")

    test("from ypyquant import BacktestEngine", test_import_backtest_engine)
    test("from ypyquant import BacktestConfig", test_import_backtest_config)
    test("from ypyquant import BacktestResult", test_import_backtest_result)
    test("from ypyquant import PerformanceMetrics, PerformanceAnalyzer", test_import_performance_metrics)
    test("from ypyquant import PortfolioOptimizationModel", test_import_portfolio_optimization)
    test("from ypyquant import WalkForwardEngine, ConfigWalkForward", test_import_walk_forward)
    test("from ypyquant import BlackLittermanQuantLib, View", test_import_black_litterman)
    test("from ypyquant import CAPMModel, FamaFrenchModel", test_import_factor_models)
    test("from ypyquant import TasaLibreRiesgoDinamica", test_import_risk_free)
    test("from ypyquant import FrenchDataFetcher", test_import_french_fetcher)
    test("from ypyquant import obtener_preset, listar_presets", test_import_presets)
    test("listar_presets() devuelve dict no vacío", test_listar_presets_returns_dict)
    test("__version__ == '0.2.0'", test_version)
    test("__all__ definido", test_all_defined)

    passed = sum(1 for r, _ in results if r == PASS)
    failed = sum(1 for r, _ in results if r == FAIL)
    print(f"\nResultado: {passed}/{len(results)} tests pasados")
    if failed:
        sys.exit(1)
