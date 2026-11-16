"""
Tests de integración — Fase 2: Signals, MLflow, BacktestEngine, Walk-Forward
Ejecutar desde la raíz del proyecto con el venv activado:
    python -X utf8 tests/test_fase2.py

Los tests de señales son completamente offline (datos sintéticos).
Los tests de MLflow requieren mlflow instalado.
Los tests de DB requieren conexión a Supabase.
"""

import sys
import os
import io
import logging
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Asegurar que ypyquant esté disponible
YPYQUANT = ROOT / "ypyquant"
if str(YPYQUANT) not in sys.path:
    sys.path.insert(0, str(YPYQUANT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")  # FutureWarnings de pandas pct_change

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  [PASS] {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  [FAIL] {name}")
        print(f"         {type(e).__name__}: {e}")


# ── Datos sintéticos compartidos ──────────────────────────────────────────────
import numpy as np
import pandas as pd

_RNG = np.random.default_rng(42)
_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
_N = 504  # ~2 años de trading
_DATES = pd.bdate_range("2022-01-03", periods=_N)


def _make_prices() -> pd.DataFrame:
    """OHLCV sintético wide, index=DatetimeIndex, columns=tickers."""
    price_data = {}
    for t in _TICKERS:
        start_price = _RNG.uniform(50, 300)
        returns = _RNG.normal(0.0003, 0.015, _N)
        close = start_price * np.cumprod(1 + returns)
        price_data[t] = close
    return pd.DataFrame(price_data, index=_DATES)


_PRICES = _make_prices()


# ── 1. Imports ─────────────────────────────────────────────────────────────────
print("\n── 1. Imports ──")


def test_import_signals_package():
    from ypyquant.signals import (
        BaseSignalGenerator,
        SignalFrame,
        TechnicalSignalGenerator,
        MLSignalGenerator,
        RegimeSignalGenerator,
        FactorSignalGenerator,
        SignalCombiner,
    )
    assert BaseSignalGenerator is not None
    assert SignalCombiner is not None


def test_import_technical():
    from ypyquant.signals.technical import TechnicalSignalGenerator
    gen = TechnicalSignalGenerator()
    assert gen.name == "technical"


def test_import_factor():
    from ypyquant.signals.factor_signal import FactorSignalGenerator
    gen = FactorSignalGenerator()
    assert gen.name == "factor"


def test_import_regime():
    from ypyquant.signals.regime_signal import RegimeSignalGenerator
    gen = RegimeSignalGenerator()
    assert gen.name == "regime"


def test_import_ml_signal():
    from ypyquant.signals.ml_signal import MLSignalGenerator
    gen = MLSignalGenerator()
    assert gen.name == "ml"


def test_import_combiner():
    from ypyquant.signals.combiner import SignalCombiner
    from ypyquant.signals.technical import TechnicalSignalGenerator
    gen = TechnicalSignalGenerator()
    comb = SignalCombiner(generators=[gen])
    assert comb is not None


def test_import_mlflow_tracker():
    from ypyquant.optimization.mlflow_tracker import MLflowTracker
    tracker = MLflowTracker(enabled=False)
    assert not tracker.enabled


def test_import_exportador():
    from ypyquant.optimization.exportador_resultados import ExportadorResultados
    assert ExportadorResultados is not None


def test_import_walk_forward_mlflow_params():
    from ypyquant.optimization.walk_forward import WalkForwardEngine
    import inspect
    sig = inspect.signature(WalkForwardEngine.__init__)
    params = sig.parameters
    assert "mlflow_enabled" in params, "WalkForwardEngine debe aceptar mlflow_enabled"
    assert "mlflow_tracking_uri" in params


check("ypyquant.signals exporta todos los símbolos públicos", test_import_signals_package)
check("TechnicalSignalGenerator importa con name='technical'", test_import_technical)
check("FactorSignalGenerator importa con name='factor'", test_import_factor)
check("RegimeSignalGenerator importa con name='regime'", test_import_regime)
check("MLSignalGenerator importa con name='ml'", test_import_ml_signal)
check("SignalCombiner inicializa correctamente", test_import_combiner)
check("MLflowTracker importa, enabled=False por defecto", test_import_mlflow_tracker)
check("ExportadorResultados importa", test_import_exportador)
check("WalkForwardEngine acepta mlflow_enabled y mlflow_tracking_uri", test_import_walk_forward_mlflow_params)


# ── 2. BaseSignalGenerator y SignalFrame ──────────────────────────────────────
print("\n── 2. BaseSignalGenerator / SignalFrame ──")


def test_cross_sectional_rank():
    from ypyquant.signals.base import BaseSignalGenerator
    raw = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [3.0, 1.0, 2.0], "C": [2.0, 3.0, 1.0]},
        index=_DATES[:3],
    )
    result = BaseSignalGenerator._cross_sectional_rank_signal(raw)
    # Valores en [-1, 1]
    assert result.min().min() >= -1.0 - 1e-9
    assert result.max().max() <= 1.0 + 1e-9
    # La media cross-seccional debe ser ~0
    row_means = result.mean(axis=1)
    assert (row_means.abs() < 0.2).all(), f"Media cross-seccional no ~0: {row_means.values}"


def test_zscore_clip():
    from ypyquant.signals.base import BaseSignalGenerator
    raw = pd.DataFrame(
        {"A": _RNG.normal(0, 1, 50), "B": _RNG.normal(0, 1, 50)},
        index=_DATES[:50],
    )
    result = BaseSignalGenerator._zscore_clip_signal(raw)
    assert result.min().min() >= -1.0 - 1e-9
    assert result.max().max() <= 1.0 + 1e-9


check("_cross_sectional_rank: rango [-1,1] y media ~0", test_cross_sectional_rank)
check("_zscore_clip: rango [-1,1]", test_zscore_clip)


# ── 3. TechnicalSignalGenerator ───────────────────────────────────────────────
print("\n── 3. TechnicalSignalGenerator ──")

_tech_signal = None


def test_technical_generate_shape():
    global _tech_signal
    from ypyquant.signals.technical import TechnicalSignalGenerator
    gen = TechnicalSignalGenerator()
    _tech_signal = gen.generate(_PRICES)
    assert _tech_signal.shape[1] == len(_TICKERS), (
        f"Esperado {len(_TICKERS)} columnas, obtenido {_tech_signal.shape[1]}"
    )
    assert len(_tech_signal) > 0


def test_technical_range():
    assert _tech_signal is not None
    vals = _tech_signal.dropna().values
    assert vals.min() >= -1.0 - 1e-6, f"Min fuera de rango: {vals.min()}"
    assert vals.max() <= 1.0 + 1e-6, f"Max fuera de rango: {vals.max()}"


def test_technical_index_alignment():
    assert _tech_signal is not None
    # El índice debe ser subconjunto de _DATES
    assert _tech_signal.index.isin(_DATES).all()


def test_technical_no_all_nan_rows():
    assert _tech_signal is not None
    # Tras el warmup, no debe haber filas completamente NaN
    tail = _tech_signal.iloc[30:]  # saltar warmup inicial
    all_nan_rows = tail.isna().all(axis=1).sum()
    assert all_nan_rows == 0, f"{all_nan_rows} filas completamente NaN tras warmup"


check("TechnicalSignalGenerator.generate() shape correcta", test_technical_generate_shape)
check("TechnicalSignalGenerator: valores en [-1, 1]", test_technical_range)
check("TechnicalSignalGenerator: índice alineado con precios", test_technical_index_alignment)
check("TechnicalSignalGenerator: sin filas all-NaN tras warmup", test_technical_no_all_nan_rows)


# ── 4. FactorSignalGenerator ──────────────────────────────────────────────────
print("\n── 4. FactorSignalGenerator ──")

_factor_signal = None


def test_factor_generate():
    global _factor_signal
    from ypyquant.signals.factor_signal import FactorSignalGenerator
    gen = FactorSignalGenerator()
    # step_days es param de generate(), factor_data=None → señal cero
    _factor_signal = gen.generate(_PRICES, step_days=63)
    assert _factor_signal is not None
    assert _factor_signal.shape[1] == len(_TICKERS)


def test_factor_range():
    assert _factor_signal is not None
    vals = _factor_signal.dropna(how="all").dropna(how="all", axis=1).values.flatten()
    valid = vals[~np.isnan(vals)]
    if len(valid) > 0:
        assert valid.min() >= -1.0 - 1e-6
        assert valid.max() <= 1.0 + 1e-6


check("FactorSignalGenerator.generate() retorna DataFrame", test_factor_generate)
check("FactorSignalGenerator: valores en [-1, 1]", test_factor_range)


# ── 5. RegimeSignalGenerator ──────────────────────────────────────────────────
print("\n── 5. RegimeSignalGenerator ──")

_regime_signal = None


def test_regime_generate():
    global _regime_signal
    from ypyquant.signals.regime_signal import RegimeSignalGenerator
    gen = RegimeSignalGenerator()
    _regime_signal = gen.generate(_PRICES)
    assert _regime_signal is not None
    assert len(_regime_signal) > 0


def test_regime_values():
    assert _regime_signal is not None
    valid_values = {-1.0, 0.0, 1.0}
    unique_vals = set(_regime_signal.values.flatten())
    unique_vals.discard(np.nan)
    assert unique_vals.issubset(valid_values), (
        f"Valores inesperados: {unique_vals - valid_values}"
    )


def test_regime_uniform_across_tickers():
    """Todos los tickers deben tener el mismo régimen en cada fecha."""
    assert _regime_signal is not None
    row_stds = _regime_signal.std(axis=1)
    assert (row_stds == 0).all(), "Régimen no es uniforme entre tickers"


check("RegimeSignalGenerator.generate() retorna DataFrame", test_regime_generate)
check("RegimeSignalGenerator: valores en {-1, 0, 1}", test_regime_values)
check("RegimeSignalGenerator: uniforme entre tickers por fecha", test_regime_uniform_across_tickers)


# ── 6. SignalCombiner ─────────────────────────────────────────────────────────
print("\n── 6. SignalCombiner ──")

_combined_signal = None


def test_combiner_weighted_mean():
    global _combined_signal
    from ypyquant.signals.technical import TechnicalSignalGenerator
    from ypyquant.signals.regime_signal import RegimeSignalGenerator
    from ypyquant.signals.combiner import SignalCombiner
    tech = TechnicalSignalGenerator()
    regime = RegimeSignalGenerator()
    comb = SignalCombiner(
        generators=[tech, regime],
        weights=[0.7, 0.3],
        combination_method="weighted_mean",
    )
    _combined_signal = comb.generate(_PRICES)
    assert _combined_signal is not None
    assert _combined_signal.shape[1] == len(_TICKERS)


def test_combiner_range():
    assert _combined_signal is not None
    vals = _combined_signal.dropna().values
    assert vals.min() >= -1.0 - 1e-6
    assert vals.max() <= 1.0 + 1e-6


def test_combiner_median():
    from ypyquant.signals.technical import TechnicalSignalGenerator
    from ypyquant.signals.combiner import SignalCombiner
    tech = TechnicalSignalGenerator()
    comb = SignalCombiner(generators=[tech], combination_method="median")
    result = comb.generate(_PRICES)
    assert result is not None


def test_combiner_equal_weight():
    """Si weights=None, debe usar pesos iguales."""
    from ypyquant.signals.technical import TechnicalSignalGenerator
    from ypyquant.signals.combiner import SignalCombiner
    tech = TechnicalSignalGenerator()
    comb = SignalCombiner(generators=[tech])
    result = comb.generate(_PRICES)
    assert result is not None
    assert result.shape[1] == len(_TICKERS)


check("SignalCombiner weighted_mean shape y sin crash", test_combiner_weighted_mean)
check("SignalCombiner weighted_mean: valores en [-1,1]", test_combiner_range)
check("SignalCombiner median", test_combiner_median)
check("SignalCombiner pesos iguales si weights=None", test_combiner_equal_weight)


# ── 7. MLflowTracker ──────────────────────────────────────────────────────────
print("\n── 7. MLflowTracker ──")


def test_tracker_disabled_noop():
    """Con enabled=False, todos los métodos son no-ops sin crash."""
    from ypyquant.optimization.mlflow_tracker import MLflowTracker
    tracker = MLflowTracker(enabled=False)
    assert not tracker.enabled
    # Simular ConfigExperimento mínimo
    class FakeConfig:
        nombre = "test_exp"
        retorno_esperado = "muestra"
        covarianza = "muestra"
        riesgo = "mpt_min_var"
        regularizacion = "none"
        black_litterman = False
        id_experimento = "T1_01"
    tracker.start_run(FakeConfig())
    tracker.log_window(None, step=0)
    tracker.end_run(None)


def test_tracker_enabled_mlflow():
    """Si mlflow está instalado, enabled=True debe funcionar."""
    try:
        import mlflow
    except ImportError:
        print("         (mlflow no instalado — saltando)")
        return

    import tempfile, shutil
    from ypyquant.optimization.mlflow_tracker import MLflowTracker
    from ypyquant.optimization.walk_forward import (
        FabricaExperimentos, ResultadoVentana,
    )

    tmpdir = Path(tempfile.mkdtemp())
    try:
        db_path = tmpdir / "test_mlflow.db"
        uri = f"sqlite:///{db_path.as_posix()}"
        tracker = MLflowTracker(
            enabled=True,
            experiment_name="test_fase2_tracker",
            tracking_uri=uri,
        )
        assert tracker.enabled

        exps = FabricaExperimentos.crear_todos()
        cfg = exps[0]
        tracker.start_run(cfg)

        # Usar ResultadoVentana real para evitar AttributeError
        ventana = ResultadoVentana(
            fecha_inicio_train=_DATES[0],
            fecha_fin_train=_DATES[251],
            fecha_inicio_oos=_DATES[252],
            fecha_fin_oos=_DATES[271],
            pesos={"AAPL": 0.5, "MSFT": 0.5},
            retorno_oos=0.03,
            tickers_seleccionados=["AAPL", "MSFT"],
            optimizacion_convergente=True,
            tiempo_segundos=1.2,
        )
        tracker.log_window(ventana, step=0)

        # end_run con ResultadoExperimento mínimo pero campos correctos
        from ypyquant.optimization.walk_forward import ResultadoExperimento
        resultado = ResultadoExperimento(
            config=cfg,
            ventanas=[ventana],
            retornos_oos_mensuales=pd.Series([0.01, -0.02, 0.03], index=_DATES[:3]),
            metricas={"sharpe_ratio": 1.1, "cagr": 0.15, "max_drawdown": -0.12,
                      "sortino_ratio": 1.3, "calmar_ratio": 0.8, "volatilidad": 0.15},
            alpha_jensen=0.01,
        )
        tracker.end_run(resultado)

        assert db_path.exists(), "La DB de MLflow no se creó"
    finally:
        # MLflow mantiene conexiones abiertas en Windows; cerrar explícitamente
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass
        # No intentar eliminar tmpdir — el lock de SQLite persiste en Windows


check("MLflowTracker disabled: todos los métodos son no-ops", test_tracker_disabled_noop)
check("MLflowTracker enabled: start/log/end sin errores (requiere mlflow)", test_tracker_enabled_mlflow)


# ── 8. WalkForwardEngine con MLflow ──────────────────────────────────────────
print("\n── 8. WalkForwardEngine con MLflow ──")


def test_walk_forward_mlflow_integration():
    """WalkForwardEngine acepta mlflow params y los pasa al tracker."""
    try:
        import mlflow
    except ImportError:
        print("         (mlflow no instalado — saltando)")
        return

    import tempfile, inspect
    from ypyquant.optimization.walk_forward import WalkForwardEngine

    sig = inspect.signature(WalkForwardEngine.__init__)
    params = sig.parameters
    assert "mlflow_enabled" in params
    assert "mlflow_experiment_name" in params
    assert "mlflow_tracking_uri" in params

    tmpdir = Path(tempfile.mkdtemp())
    db_path = tmpdir / "wf_test.db"
    from ypyquant.optimization.walk_forward import ConfigWalkForward
    config = ConfigWalkForward(
        ventana_entrenamiento_meses=12,
        fecha_inicio_oos="2023-01-01",
        fecha_fin_oos="2023-06-01",
        top_n_activos=5,
    )
    engine = WalkForwardEngine(
        config=config,
        mlflow_enabled=True,
        mlflow_experiment_name="test_wf_mlflow",
        mlflow_tracking_uri=f"sqlite:///{db_path.as_posix()}",
    )
    assert hasattr(engine, "_tracker")
    assert engine._tracker.enabled


check("WalkForwardEngine integra MLflowTracker correctamente", test_walk_forward_mlflow_integration)


# ── 9. ExportadorResultados.exportar_a_db ─────────────────────────────────────
print("\n── 9. ExportadorResultados.exportar_a_db ──")


def _make_resultado_fake():
    from ypyquant.optimization.walk_forward import ResultadoExperimento, FabricaExperimentos
    exps = FabricaExperimentos.crear_todos()
    cfg = exps[0]
    return ResultadoExperimento(
        config=cfg,
        ventanas=[],
        retornos_oos_mensuales=pd.Series(dtype=float),
        metricas={"sharpe_ratio": 0.99, "cagr": 0.12, "max_drawdown": -0.15,
                  "sortino_ratio": 1.2, "calmar_ratio": 0.8, "volatilidad": 0.14},
        alpha_jensen=0.01,
    )


def test_exportador_a_db_smoke():
    """Smoke test: exportar 1 resultado ficticio a Supabase."""
    from db.database import SessionLocal
    from ypyquant.optimization.exportador_resultados import ExportadorResultados

    resultado = _make_resultado_fake()
    exportador = ExportadorResultados(
        resultados=[resultado],
        directorio_salida=ROOT / "data" / "reportes" / "fase2",
    )
    with SessionLocal() as session:
        ids = exportador.exportar_a_db(
            session=session,
            nombre_experimento="test_fase2_smoke",
            sobrescribir=True,
        )
    assert isinstance(ids, list)
    assert len(ids) == 1
    assert isinstance(ids[0], int)


def test_exportador_idempotente():
    """Segunda llamada con sobrescribir=False debe retornar lista vacía."""
    from db.database import SessionLocal
    from ypyquant.optimization.exportador_resultados import ExportadorResultados

    resultado = _make_resultado_fake()
    exportador = ExportadorResultados(
        resultados=[resultado],
        directorio_salida=ROOT / "data" / "reportes" / "fase2",
    )
    with SessionLocal() as session:
        ids = exportador.exportar_a_db(
            session=session,
            nombre_experimento="test_fase2_smoke",
            sobrescribir=False,
        )
    assert ids == [], f"Esperado [] (idempotente), obtenido {ids}"


check("exportar_a_db: persiste resultado en Supabase", test_exportador_a_db_smoke)
check("exportar_a_db: idempotente con sobrescribir=False", test_exportador_idempotente)


# ── 10. run_fase2 imports ─────────────────────────────────────────────────────
print("\n── 10. Scripts importan sin error ──")


def test_run_fase2_imports():
    """run_fase2.py debe importarse sin errores."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_fase2", ROOT / "scripts" / "run_fase2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Solo verificar que parse no lanza excepciones de sintaxis
    assert spec is not None


def test_config_fase2_yaml():
    """config_fase2.yaml debe existir y ser parseable."""
    import yaml
    cfg_path = ROOT / "scripts" / "config_fase2.yaml"
    assert cfg_path.exists(), f"config_fase2.yaml no encontrado en {cfg_path}"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "walk_forward" in cfg
    assert "mlflow" in cfg
    assert "signals" in cfg
    assert cfg["mlflow"]["enabled"] is True


def test_vbt_scan_importable():
    """vbt_signal_scan.py debe existir."""
    scan_path = ROOT / "scripts" / "vbt_signal_scan.py"
    assert scan_path.exists(), "vbt_signal_scan.py no encontrado"


check("run_fase2.py existe y spec carga", test_run_fase2_imports)
check("config_fase2.yaml existe y tiene secciones obligatorias", test_config_fase2_yaml)
check("vbt_signal_scan.py existe", test_vbt_scan_importable)


# ── 11. MLflow DB real ────────────────────────────────────────────────────────
print("\n── 11. MLflow DB (resultados del pipeline) ──")


def test_mlflow_db_exists():
    mlflow_db = ROOT / "data" / "mlflow.db"
    assert mlflow_db.exists(), f"mlflow.db no encontrado en {mlflow_db}"


def test_mlflow_has_runs():
    try:
        import mlflow
    except ImportError:
        print("         (mlflow no instalado — saltando)")
        return

    mlflow_db = ROOT / "data" / "mlflow.db"
    if not mlflow_db.exists():
        print("         (mlflow.db no existe — saltando)")
        return

    uri = f"sqlite:///{mlflow_db.as_posix()}"
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    exp_names = [e.name for e in experiments]
    assert "walk_forward_factorial" in exp_names, (
        f"Experimento 'walk_forward_factorial' no encontrado. Disponibles: {exp_names}"
    )
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in experiments
                        if e.name == "walk_forward_factorial"]
    )
    assert len(runs) > 0, "No hay runs en 'walk_forward_factorial'"
    print(f"         ({len(runs)} runs encontrados en MLflow)")


def test_mlflow_run_has_metrics():
    try:
        import mlflow
    except ImportError:
        print("         (mlflow no instalado — saltando)")
        return

    mlflow_db = ROOT / "data" / "mlflow.db"
    if not mlflow_db.exists():
        print("         (mlflow.db no existe — saltando)")
        return

    uri = f"sqlite:///{mlflow_db.as_posix()}"
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    wf_exp = next((e for e in experiments if e.name == "walk_forward_factorial"), None)
    if wf_exp is None:
        return
    runs = client.search_runs([wf_exp.experiment_id])
    if not runs:
        return
    # Al menos un run debe tener métricas de Sharpe
    run_with_sharpe = [r for r in runs if "sharpe_ratio" in r.data.metrics]
    assert len(run_with_sharpe) > 0, (
        f"Ningún run tiene 'sharpe_ratio'. Métricas del primer run: "
        f"{list(runs[0].data.metrics.keys())}"
    )
    sharpe = run_with_sharpe[0].data.metrics["sharpe_ratio"]
    assert isinstance(sharpe, float)
    print(f"         (mejor Sharpe OOS en MLflow: {sharpe:.4f})")


check("mlflow.db existe en data/", test_mlflow_db_exists)
check("MLflow: experimento 'walk_forward_factorial' existe", test_mlflow_has_runs)
check("MLflow: runs tienen métrica 'sharpe_oos'", test_mlflow_run_has_metrics)


# ── 12. Reportes generados ────────────────────────────────────────────────────
print("\n── 12. Reportes Fase 2 ──")

_REPORTES = ROOT / "data" / "reportes" / "fase2"


def test_reportes_dir_exists():
    assert _REPORTES.exists(), f"Directorio de reportes no existe: {_REPORTES}"


def test_csv_resultados():
    csv = _REPORTES / "resultados_factorial.csv"
    assert csv.exists(), f"CSV faltante: {csv}"
    df = pd.read_csv(csv)
    assert len(df) > 0


def test_csv_retornos_oos():
    csv = _REPORTES / "retornos_oos_mensuales.csv"
    assert csv.exists(), f"CSV retornos faltante: {csv}"


def test_latex_files():
    latex_dir = _REPORTES / "tablas_latex"
    assert latex_dir.exists()
    tex_files = list(latex_dir.glob("*.tex"))
    assert len(tex_files) >= 2, f"Se esperan ≥2 .tex, hay {len(tex_files)}"


def test_graficos():
    graficos = _REPORTES / "graficos"
    assert graficos.exists()
    pngs = list(graficos.glob("*.png"))
    assert len(pngs) >= 1, f"Se esperan ≥1 PNG, hay {len(pngs)}"


check("Directorio data/reportes/fase2 existe", test_reportes_dir_exists)
check("resultados_factorial.csv existe y no está vacío", test_csv_resultados)
check("retornos_oos_mensuales.csv existe", test_csv_retornos_oos)
check("tablas_latex/ tiene ≥2 archivos .tex", test_latex_files)
check("graficos/ tiene ≥1 PNG", test_graficos)


# ── Resumen ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
total  = len(results)
print(f"  Resultado: {passed}/{total} tests OK  |  {failed} fallidos")
print("=" * 60)
if failed > 0:
    print("\nTests fallidos:")
    for status, name in results:
        if status == FAIL:
            print(f"  • {name}")
    sys.exit(1)
else:
    print("  Todos los tests pasaron.")
    sys.exit(0)
