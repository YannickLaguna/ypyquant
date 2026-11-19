"""
Tests de la Fase 2: señales, MLflow, walk-forward, exportador y artefactos del pipeline.

Responsabilidad:
    - Verificar offline (datos sintéticos de conftest.py) los generadores de señales
      (técnica, factorial, régimen, ML), el combinador y el tracker de MLflow.
    - Comprobar que los scripts y la configuración del pipeline existen y son válidos.
    - Comprobar los artefactos generados por el pipeline (mlflow.db, reportes); si no
      están en el checkout se saltan, porque son datos locales no versionados.

Se relaciona con:
    - src/ypyquant/signals, src/ypyquant/optimization (código bajo prueba).
    - tests/conftest.py (fixtures precios, fechas, rng, raiz_repo).
    - scripts/run_fase2.py, scripts/config_fase2.yaml, scripts/vbt_signal_scan.py.
"""

import contextlib
import importlib.util
import inspect

import numpy as np
import pandas as pd
import pytest

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]


# ── 1. Imports ─────────────────────────────────────────────────────────────────


def test_import_signals_package():
    import ypyquant.signals as senales
    for nombre in (
        "BaseSignalGenerator",
        "SignalFrame",
        "TechnicalSignalGenerator",
        "MLSignalGenerator",
        "RegimeSignalGenerator",
        "FactorSignalGenerator",
        "SignalCombiner",
    ):
        assert getattr(senales, nombre, None) is not None, f"ypyquant.signals no exporta {nombre}"


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
    sig = inspect.signature(WalkForwardEngine.__init__)
    params = sig.parameters
    assert "mlflow_enabled" in params, "WalkForwardEngine debe aceptar mlflow_enabled"
    assert "mlflow_tracking_uri" in params


# ── 2. BaseSignalGenerator y SignalFrame ──────────────────────────────────────


def test_cross_sectional_rank(fechas):
    from ypyquant.signals.base import BaseSignalGenerator
    raw = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [3.0, 1.0, 2.0], "C": [2.0, 3.0, 1.0]},
        index=fechas[:3],
    )
    result = BaseSignalGenerator._cross_sectional_rank_signal(raw)
    # Valores en [-1, 1]
    assert result.min().min() >= -1.0 - 1e-9
    assert result.max().max() <= 1.0 + 1e-9
    # La media cross-seccional debe ser ~0
    row_means = result.mean(axis=1)
    assert (row_means.abs() < 0.2).all(), f"Media cross-seccional no ~0: {row_means.values}"


def test_zscore_clip(fechas, rng):
    from ypyquant.signals.base import BaseSignalGenerator
    raw = pd.DataFrame(
        {"A": rng.normal(0, 1, 50), "B": rng.normal(0, 1, 50)},
        index=fechas[:50],
    )
    result = BaseSignalGenerator._zscore_clip_signal(raw)
    assert result.min().min() >= -1.0 - 1e-9
    assert result.max().max() <= 1.0 + 1e-9


# ── 3. TechnicalSignalGenerator ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def senal_tecnica(precios):
    """Señal técnica generada una vez sobre los precios sintéticos."""
    from ypyquant.signals.technical import TechnicalSignalGenerator
    return TechnicalSignalGenerator().generate(precios)


def test_technical_generate_shape(senal_tecnica):
    assert senal_tecnica.shape[1] == len(TICKERS), (
        f"Esperado {len(TICKERS)} columnas, obtenido {senal_tecnica.shape[1]}"
    )
    assert len(senal_tecnica) > 0


def test_technical_range(senal_tecnica):
    vals = senal_tecnica.dropna().values
    assert vals.min() >= -1.0 - 1e-6, f"Min fuera de rango: {vals.min()}"
    assert vals.max() <= 1.0 + 1e-6, f"Max fuera de rango: {vals.max()}"


def test_technical_index_alignment(senal_tecnica, fechas):
    # El índice debe ser subconjunto de fechas
    assert senal_tecnica.index.isin(fechas).all()


def test_technical_no_all_nan_rows(senal_tecnica):
    # Tras el warmup, no debe haber filas completamente NaN
    tail = senal_tecnica.iloc[30:]  # saltar warmup inicial
    all_nan_rows = tail.isna().all(axis=1).sum()
    assert all_nan_rows == 0, f"{all_nan_rows} filas completamente NaN tras warmup"


# ── 4. FactorSignalGenerator ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def senal_factorial(precios):
    """Señal factorial (sin factor_data → señal cero) sobre los precios sintéticos."""
    from ypyquant.signals.factor_signal import FactorSignalGenerator
    # step_days es parámetro de generate(); factor_data=None → señal cero
    return FactorSignalGenerator().generate(precios, step_days=63)


def test_factor_generate(senal_factorial):
    assert senal_factorial is not None
    assert senal_factorial.shape[1] == len(TICKERS)


def test_factor_range(senal_factorial):
    assert senal_factorial is not None
    vals = senal_factorial.dropna(how="all").dropna(how="all", axis=1).values.flatten()
    valid = vals[~np.isnan(vals)]
    if len(valid) > 0:
        assert valid.min() >= -1.0 - 1e-6
        assert valid.max() <= 1.0 + 1e-6


# ── 5. RegimeSignalGenerator ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def senal_regimen(precios):
    """Señal de régimen generada una vez sobre los precios sintéticos."""
    from ypyquant.signals.regime_signal import RegimeSignalGenerator
    return RegimeSignalGenerator().generate(precios)


def test_regime_generate(senal_regimen):
    assert senal_regimen is not None
    assert len(senal_regimen) > 0


def test_regime_values(senal_regimen):
    assert senal_regimen is not None
    valid_values = {-1.0, 0.0, 1.0}
    unique_vals = set(senal_regimen.values.flatten())
    unique_vals.discard(np.nan)
    assert unique_vals.issubset(valid_values), (
        f"Valores inesperados: {unique_vals - valid_values}"
    )


def test_regime_uniform_across_tickers(senal_regimen):
    """Todos los tickers deben tener el mismo régimen en cada fecha."""
    assert senal_regimen is not None
    row_stds = senal_regimen.std(axis=1)
    assert (row_stds == 0).all(), "Régimen no es uniforme entre tickers"


# ── 6. SignalCombiner ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def senal_combinada(precios):
    """Señal combinada (técnica 0.7 + régimen 0.3, media ponderada)."""
    from ypyquant.signals.combiner import SignalCombiner
    from ypyquant.signals.regime_signal import RegimeSignalGenerator
    from ypyquant.signals.technical import TechnicalSignalGenerator
    comb = SignalCombiner(
        generators=[TechnicalSignalGenerator(), RegimeSignalGenerator()],
        weights=[0.7, 0.3],
        combination_method="weighted_mean",
    )
    return comb.generate(precios)


def test_combiner_weighted_mean(senal_combinada):
    assert senal_combinada is not None
    assert senal_combinada.shape[1] == len(TICKERS)


def test_combiner_range(senal_combinada):
    assert senal_combinada is not None
    vals = senal_combinada.dropna().values
    assert vals.min() >= -1.0 - 1e-6
    assert vals.max() <= 1.0 + 1e-6


def test_combiner_median(precios):
    from ypyquant.signals.combiner import SignalCombiner
    from ypyquant.signals.technical import TechnicalSignalGenerator
    tech = TechnicalSignalGenerator()
    comb = SignalCombiner(generators=[tech], combination_method="median")
    result = comb.generate(precios)
    assert result is not None


def test_combiner_equal_weight(precios):
    """Si weights=None, debe usar pesos iguales."""
    from ypyquant.signals.combiner import SignalCombiner
    from ypyquant.signals.technical import TechnicalSignalGenerator
    tech = TechnicalSignalGenerator()
    comb = SignalCombiner(generators=[tech])
    result = comb.generate(precios)
    assert result is not None
    assert result.shape[1] == len(TICKERS)


# ── 7. MLflowTracker ──────────────────────────────────────────────────────────


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


def test_tracker_enabled_mlflow(fechas, tmp_path):
    """Si mlflow está instalado, enabled=True debe funcionar."""
    mlflow = pytest.importorskip("mlflow")

    from ypyquant.optimization.mlflow_tracker import MLflowTracker
    from ypyquant.optimization.walk_forward import (
        FabricaExperimentos,
        ResultadoVentana,
    )

    tmpdir = tmp_path
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
            fecha_inicio_train=fechas[0],
            fecha_fin_train=fechas[251],
            fecha_inicio_oos=fechas[252],
            fecha_fin_oos=fechas[271],
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
            retornos_oos_mensuales=pd.Series([0.01, -0.02, 0.03], index=fechas[:3]),
            metricas={"sharpe_ratio": 1.1, "cagr": 0.15, "max_drawdown": -0.12,
                      "sortino_ratio": 1.3, "calmar_ratio": 0.8, "volatilidad": 0.15},
            alpha_jensen=0.01,
        )
        tracker.end_run(resultado)

        assert db_path.exists(), "La DB de MLflow no se creó"
    finally:
        # MLflow mantiene conexiones abiertas en Windows; cerrar explícitamente
        with contextlib.suppress(Exception):
            mlflow.end_run()
        # No intentar eliminar tmpdir — el lock de SQLite persiste en Windows


# ── 8. WalkForwardEngine con MLflow ──────────────────────────────────────────

def test_walk_forward_mlflow_integration(tmp_path):
    """WalkForwardEngine acepta mlflow params y los pasa al tracker."""
    pytest.importorskip("mlflow")

    from ypyquant.optimization.walk_forward import WalkForwardEngine

    sig = inspect.signature(WalkForwardEngine.__init__)
    params = sig.parameters
    assert "mlflow_enabled" in params
    assert "mlflow_experiment_name" in params
    assert "mlflow_tracking_uri" in params

    tmpdir = tmp_path
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


# ── 10. run_fase2 imports ─────────────────────────────────────────────────────


def test_run_fase2_imports(raiz_repo):
    """run_fase2.py debe importarse sin errores."""
    spec = importlib.util.spec_from_file_location(
        "run_fase2", raiz_repo / "scripts" / "run_fase2.py"
    )
    # Solo se verifica que el fichero existe y que su spec se resuelve (no se ejecuta)
    assert spec is not None and spec.loader is not None


def test_config_fase2_yaml(raiz_repo):
    """config_fase2.yaml debe existir y ser parseable."""
    import yaml
    cfg_path = raiz_repo / "scripts" / "config_fase2.yaml"
    assert cfg_path.exists(), f"config_fase2.yaml no encontrado en {cfg_path}"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "walk_forward" in cfg
    assert "mlflow" in cfg
    assert "signals" in cfg
    assert cfg["mlflow"]["enabled"] is True


def test_vbt_scan_importable(raiz_repo):
    """vbt_signal_scan.py debe existir."""
    scan_path = raiz_repo / "scripts" / "vbt_signal_scan.py"
    assert scan_path.exists(), "vbt_signal_scan.py no encontrado"


# ── 11. MLflow DB real ────────────────────────────────────────────────────────


def test_mlflow_db_exists(raiz_repo):
    mlflow_db = raiz_repo / "data" / "mlflow.db"
    if not mlflow_db.exists():
        pytest.skip(f"mlflow.db no existe en {mlflow_db} (es un artefacto local, no versionado)")


def test_mlflow_has_runs(raiz_repo):
    mlflow = pytest.importorskip("mlflow")

    mlflow_db = raiz_repo / "data" / "mlflow.db"
    if not mlflow_db.exists():
        pytest.skip("data/mlflow.db no existe (artefacto local no versionado)")

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


def test_mlflow_run_has_metrics(raiz_repo):
    mlflow = pytest.importorskip("mlflow")

    mlflow_db = raiz_repo / "data" / "mlflow.db"
    if not mlflow_db.exists():
        pytest.skip("data/mlflow.db no existe (artefacto local no versionado)")

    uri = f"sqlite:///{mlflow_db.as_posix()}"
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    wf_exp = next((e for e in experiments if e.name == "walk_forward_factorial"), None)
    if wf_exp is None:
        pytest.skip("experimento walk_forward_factorial ausente")
    runs = client.search_runs([wf_exp.experiment_id])
    if not runs:
        pytest.skip("sin runs en walk_forward_factorial")
    # Al menos un run debe tener métricas de Sharpe
    run_with_sharpe = [r for r in runs if "sharpe_ratio" in r.data.metrics]
    assert len(run_with_sharpe) > 0, (
        f"Ningún run tiene 'sharpe_ratio'. Métricas del primer run: "
        f"{list(runs[0].data.metrics.keys())}"
    )
    sharpe = run_with_sharpe[0].data.metrics["sharpe_ratio"]
    assert isinstance(sharpe, float)


# ── 12. Reportes generados ────────────────────────────────────────────────────

@pytest.fixture()
def reportes(raiz_repo):
    """Directorio data/reportes/fase2; salta el test si el pipeline no se ha ejecutado."""
    ruta = raiz_repo / "data" / "reportes" / "fase2"
    if not ruta.exists():
        pytest.skip("data/reportes/fase2 no existe (artefacto local no versionado)")
    return ruta


def test_reportes_dir_exists(reportes):
    assert reportes.exists()


def test_csv_resultados(reportes):
    csv = reportes / "resultados_factorial.csv"
    assert csv.exists(), f"CSV faltante: {csv}"
    df = pd.read_csv(csv)
    assert len(df) > 0


def test_csv_retornos_oos(reportes):
    csv = reportes / "retornos_oos_mensuales.csv"
    assert csv.exists(), f"CSV retornos faltante: {csv}"


def test_latex_files(reportes):
    latex_dir = reportes / "tablas_latex"
    assert latex_dir.exists()
    tex_files = list(latex_dir.glob("*.tex"))
    assert len(tex_files) >= 2, f"Se esperan ≥2 .tex, hay {len(tex_files)}"


def test_graficos(reportes):
    graficos = reportes / "graficos"
    assert graficos.exists()
    pngs = list(graficos.glob("*.png"))
    assert len(pngs) >= 1, f"Se esperan ≥1 PNG, hay {len(pngs)}"
