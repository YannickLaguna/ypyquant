"""
run_fase2.py — Orquestador del pipeline de research Fase 2.
===========================================================

Pipeline completo (stages 1-6):

  [1] DATA LAYER
      Supabase → cargar precios (activos) + factores (factor_data)

  [2] SIGNAL GENERATION
      SignalCombiner.generate() → SignalFrame

  [3] FAST SIGNAL SCAN  (VectorBT, opcional)
      scan_and_rank() → ranking de tickers/configs por Sharpe bruto

  [4] PORTFOLIO VALIDATION  (backtest_engine.py)
      Valida top-N configs con costes realistas

  [5] WALK-FORWARD OOS  (walk_forward.py)
      21 experimentos OOS con MLflow tracking opcional

  [6] PERSISTENCIA
      [6a] ExportadorResultados.exportar_a_db()  → Supabase tabla backtests
      [6b] Señales del mejor experimento        → Supabase tabla signals (opcional)

Uso:
    python scripts/run_fase2.py
    python scripts/run_fase2.py --config scripts/config_fase2.yaml
    python scripts/run_fase2.py --experiments 1,2,3   # subset para testing
    python scripts/run_fase2.py --skip-vbt             # omitir VectorBT scan
    python scripts/run_fase2.py --no-mlflow            # desactivar tracking
    python scripts/run_fase2.py --no-db                # no persistir en DB
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Setup paths: añadir raíz del proyecto al sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    """Carga config_fase2.yaml."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML no instalado. Ejecuta: pip install pyyaml")
        sys.exit(1)
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config no encontrado: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config cargado: {path}")
    return cfg


def _load_universe(universo_file: str) -> List[str]:
    p = _ROOT / universo_file
    if not p.exists():
        logger.warning(f"Archivo de universo no encontrado: {p}")
        return []
    tickers = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    logger.info(f"Universo: {len(tickers)} tickers")
    return tickers


_MIN_ROWS_THRESHOLD = 500  # mínimo de filas para considerar datos completos (~2 años)


def _load_prices_and_factors(
    tickers: List[str],
    start: str,
    end: str,
) -> tuple:
    """
    Carga precios y factores desde Supabase.
    Auto-fetch de tickers ausentes o con datos insuficientes.
    """
    from data.loader import load_factors
    from db.database import SessionLocal

    prices, loaded = _load_prices_from_db(tickers, start, end)

    # Detectar tickers sin datos o con pocas filas
    missing = [t for t in tickers if t not in loaded]
    thin    = [t for t in loaded if prices[t].count() < _MIN_ROWS_THRESHOLD]
    to_fetch = list(set(missing + thin))

    if to_fetch:
        logger.info(
            f"[Stage 1] Auto-fetch: {len(missing)} ausentes, {len(thin)} con datos insuficientes "
            f"(<{_MIN_ROWS_THRESHOLD} filas). Descargando {len(to_fetch)} tickers ..."
        )
        _autofetch_prices(to_fetch, start, end)
        prices, loaded = _load_prices_from_db(tickers, start, end)

    if prices.empty:
        logger.error("[Stage 1] Sin precios tras el auto-fetch. Verifica API keys en .env")
        return pd.DataFrame(), None

    logger.info(f"[Stage 1] Precios: {prices.shape} ({len(loaded)} tickers)")

    # Factores FF
    factors = None
    try:
        with SessionLocal() as session:
            factors = load_factors(start_date=start, end_date=end, session=session)
        if not factors.empty:
            if "MKT_RF" in factors.columns and "MKT" not in factors.columns:
                factors = factors.rename(columns={"MKT_RF": "MKT"})
            logger.info(f"[Stage 1] Factores FF: {factors.shape}")
        else:
            logger.warning("[Stage 1] Sin factores FF en DB (opcionales para señales técnicas).")
            factors = None
    except Exception as e:
        logger.warning(f"[Stage 1] No se pudieron cargar factores: {e}")

    return prices, factors


def _load_prices_from_db(tickers: List[str], start: str, end: str):
    """
    Carga precios desde DB.
    Returns (wide_df, set_of_loaded_tickers).
    """
    from data.loader import load_prices
    from db.database import SessionLocal

    with SessionLocal() as session:
        prices_dict = load_prices(tickers, start_date=start, end_date=end, session=session)

    if not prices_dict:
        return pd.DataFrame(), set()

    series_list = []
    for ticker, df in prices_dict.items():
        col = "Adjusted_Close" if "Adjusted_Close" in df.columns else "Close"
        series_list.append(df[col].rename(ticker))

    prices = pd.concat(series_list, axis=1).sort_index()
    prices = prices.dropna(how="all", axis=1)
    return prices, set(prices.columns)


def _autofetch_prices(tickers: List[str], start: str, end: str, delay: float = 0.5) -> None:
    """Descarga y persiste precios con triple fallback (yfinance → TwelveData → AlphaVantage)."""
    import time as _time
    from data.fetcher import DataFetcher
    from db.database import SessionLocal

    fetcher = DataFetcher()
    ok, errors = 0, []
    n = len(tickers)

    with SessionLocal() as session:
        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 1 or n <= 20:
                logger.info(f"[AutoFetch] [{i}/{n}] {ticker} ...")
            try:
                rows = fetcher.fetch_and_save(ticker, start, end, session)
                if rows and rows > 0:
                    ok += 1
            except Exception as exc:
                errors.append(ticker)
                logger.debug(f"[AutoFetch] {ticker}: {exc}")
            if i < n:
                _time.sleep(delay)

    logger.info(f"[AutoFetch] Completado: {ok}/{n} OK, {len(errors)} errores.")
    if errors:
        logger.warning(f"[AutoFetch] Fallaron: {errors[:10]}{'...' if len(errors) > 10 else ''}")


# ---------------------------------------------------------------------------
# Stage 2: Signal Generation
# ---------------------------------------------------------------------------

def run_stage2_signals(
    prices: pd.DataFrame,
    factors: Optional[pd.DataFrame],
    signal_cfg: dict,
) -> pd.DataFrame:
    """Genera SignalFrame usando SignalCombiner según configuración."""
    from ypyquant.signals import (
        TechnicalSignalGenerator,
        FactorSignalGenerator,
        RegimeSignalGenerator,
        SignalCombiner,
    )

    gen_config = signal_cfg.get("generators", [{"name": "technical", "weight": 1.0}])
    combination_method = signal_cfg.get("combination_method", "weighted_mean")

    name_to_class = {
        "technical": TechnicalSignalGenerator,
        "factor":    FactorSignalGenerator,
        "regime":    RegimeSignalGenerator,
    }

    generators = []
    weights = []
    for cfg in gen_config:
        name = cfg["name"]
        cls = name_to_class.get(name)
        if cls is None:
            logger.warning(f"[Stage 2] Generador desconocido: '{name}', omitiendo.")
            continue
        generators.append(cls())
        weights.append(float(cfg.get("weight", 1.0)))

    if not generators:
        logger.error("[Stage 2] Sin generadores válidos.")
        return pd.DataFrame(index=prices.index, columns=prices.columns)

    combiner = SignalCombiner(generators=generators, weights=weights,
                              combination_method=combination_method)
    kwargs = {}
    if factors is not None:
        kwargs["factor_data"] = factors

    logger.info(f"[Stage 2] Generando señales con {combiner.name} ...")
    signal_frame = combiner.generate(prices, **kwargs)
    logger.info(f"[Stage 2] SignalFrame: {signal_frame.shape}, rango [{signal_frame.min().min():.2f}, {signal_frame.max().max():.2f}]")
    return signal_frame


# ---------------------------------------------------------------------------
# Stage 3: VectorBT scan
# ---------------------------------------------------------------------------

def run_stage3_vbt(
    prices: pd.DataFrame,
    signal_frame: pd.DataFrame,
    vbt_cfg: dict,
) -> Optional[pd.DataFrame]:
    """Scan rápido con VectorBT, devuelve ranking de tickers."""
    if not vbt_cfg.get("enabled", True):
        logger.info("[Stage 3] VectorBT scan desactivado por config.")
        return None

    try:
        from scripts.vbt_signal_scan import scan_and_rank
    except ImportError:
        try:
            sys.path.insert(0, str(_ROOT / "scripts"))
            from vbt_signal_scan import scan_and_rank
        except ImportError as e:
            logger.warning(f"[Stage 3] No se pudo importar vbt_signal_scan: {e}")
            return None

    try:
        ranking = scan_and_rank(
            prices_df=prices,
            signal_frame=signal_frame,
            top_n=vbt_cfg.get("top_n_signals", 10),
            commission=vbt_cfg.get("commission", 0.001),
            slippage=vbt_cfg.get("slippage", 0.001),
            entry_threshold=vbt_cfg.get("entry_threshold", 0.5),
            exit_threshold=-vbt_cfg.get("entry_threshold", 0.5),
        )
        logger.info(f"[Stage 3] VBT ranking:\n{ranking.to_string()}")
        return ranking
    except Exception as e:
        logger.warning(f"[Stage 3] VectorBT scan falló: {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 4: Portfolio validation (backtest_engine)
# ---------------------------------------------------------------------------

def run_stage4_backtest(
    prices: pd.DataFrame,
    signal_frame: pd.DataFrame,
    backtest_cfg: dict,
) -> Optional[object]:
    """Validación de señales con backtest_engine (costes realistas)."""
    if not backtest_cfg.get("enabled", True):
        logger.info("[Stage 4] Backtest engine desactivado por config.")
        return None

    try:
        from ypyquant import BacktestEngine, BacktestConfig

        start_date = str(prices.index.min().date())
        end_date   = str(prices.index.max().date())

        bt_config = BacktestConfig(
            fecha_inicio=start_date,
            fecha_fin=end_date,
            capital_inicial=backtest_cfg.get("initial_capital", 100_000.0),
            frecuencia_rebalanceo="monthly",
            commission_rate=backtest_cfg.get("commission", 0.001),
            slippage_pct=backtest_cfg.get("slippage", 0.001),
        )
        engine = BacktestEngine(config=bt_config)

        # Construir Dict[ticker, OHLCV_df] que espera BacktestEngine
        historical_data = {}
        for col in prices.columns:
            df = prices[[col]].rename(columns={col: "Close"})
            df["Open"] = df["High"] = df["Low"] = df["Close"]
            df["Volume"] = 1_000_000
            historical_data[col] = df

        # Estrategia simple: pesos proporcionales a señal positiva
        def strategy_func(fecha, data):
            row = signal_frame.loc[signal_frame.index <= fecha]
            if row.empty:
                return {}
            latest = row.iloc[-1]
            pos = latest[latest > 0.3]
            total = pos.sum()
            if total <= 0:
                return {}
            return (pos / total).to_dict()

        logger.info("[Stage 4] Ejecutando backtest con costes realistas ...")
        result = engine.run_backtest(strategy_func, historical_data)
        logger.info(
            f"[Stage 4] Backtest: "
            f"Retorno={result.retorno_total_pct:.1%}, "
            f"Sharpe={result.sharpe_ratio:.3f}, "
            f"MaxDD={result.max_drawdown_pct:.1%}"
        )
        return result
    except Exception as e:
        logger.warning(f"[Stage 4] Backtest engine fallo: {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 5: Walk-forward OOS
# ---------------------------------------------------------------------------

def run_stage5_walkforward(
    prices: pd.DataFrame,
    factors: Optional[pd.DataFrame],
    wf_cfg: dict,
    mlflow_cfg: dict,
    experimentos_subset: Optional[List[int]],
) -> Dict:
    """Ejecuta 21 experimentos walk-forward con tracking MLflow."""
    from ypyquant import WalkForwardEngine, ConfigWalkForward
    from ypyquant.optimization.walk_forward import FabricaExperimentos

    # Resolver URIs relativas a ruta absoluta
    tracking_uri = mlflow_cfg.get("tracking_uri", "")
    if tracking_uri.startswith("sqlite:///data/"):
        db_rel = tracking_uri[len("sqlite:///"):]          # "data/mlflow.db"
        tracking_uri = f"sqlite:///{(_ROOT / db_rel).as_posix()}"
    elif tracking_uri.startswith("file:///data/"):
        tracking_uri = f"file:///{(_ROOT / tracking_uri[8:]).as_posix()}"

    engine = WalkForwardEngine(
        config=ConfigWalkForward(
            ventana_entrenamiento_meses=wf_cfg.get("ventana_entrenamiento_meses", 12),
            fecha_inicio_oos=wf_cfg.get("fecha_inicio_oos", "2010-01-01"),
            fecha_fin_oos=wf_cfg.get("fecha_fin_oos", "2025-01-01"),
            top_n_activos=wf_cfg.get("top_n_activos", 50),
        ),
        mlflow_enabled=mlflow_cfg.get("enabled", False),
        mlflow_experiment_name=mlflow_cfg.get("experiment_name", "walk_forward_factorial"),
        mlflow_tracking_uri=tracking_uri or None,
    )

    experimentos = FabricaExperimentos.crear_todos()
    if experimentos_subset:
        # índices 1-based → posiciones en la lista
        experimentos = [
            experimentos[i - 1]
            for i in experimentos_subset
            if 1 <= i <= len(experimentos)
        ]
        logger.info(f"[Stage 5] Subset: {[e.id_experimento for e in experimentos]}")
    else:
        logger.info(f"[Stage 5] Ejecutando {len(experimentos)} experimentos completos.")

    market_returns = None
    if factors is not None and "MKT" in factors.columns:
        market_returns = factors["MKT"]

    resultados = {}
    for config_exp in experimentos:
        try:
            resultado = engine.ejecutar_experimento(
                config_exp=config_exp,
                precios_universo=prices,
                factores_ff=factors,
                retornos_mercado=market_returns,
            )
            resultados[config_exp.id_experimento] = resultado
            sharpe = resultado.metricas.get("sharpe_ratio", "N/A")
            logger.info(f"[Stage 5] {config_exp.nombre}: Sharpe={sharpe}")
        except Exception as e:
            logger.error(f"[Stage 5] Experimento {config_exp.id_experimento} falló: {e}")

    logger.info(f"[Stage 5] {len(resultados)}/{len(experimentos)} experimentos completados.")
    return resultados


# ---------------------------------------------------------------------------
# Stage 6: Persistencia
# ---------------------------------------------------------------------------

def run_stage6_persist(
    resultados: Dict,
    prices: pd.DataFrame,
    signal_frame: pd.DataFrame,
    output_cfg: dict,
    nombre_experimento: str,
) -> None:
    """Exporta resultados a CSV/LaTeX/PNG, persistencia en Supabase."""
    from ypyquant.optimization.exportador_resultados import ExportadorResultados

    out_dir = _ROOT / output_cfg.get("directorio_salida", "data/reportes/fase2")
    exportador = ExportadorResultados(str(out_dir), resultados)

    logger.info("[Stage 6] Exportando CSV, LaTeX y gráficos ...")
    try:
        exportador.exportar_todo()
    except Exception as e:
        logger.warning(f"[Stage 6] Error en exportar_todo: {e}")

    # DB persistence
    if output_cfg.get("persistir_db", True):
        try:
            from db.database import SessionLocal
            with SessionLocal() as session:
                ids = exportador.exportar_a_db(session, nombre_experimento, sobrescribir=False)
                logger.info(f"[Stage 6a] {len(ids)} backtests persistidos en Supabase.")
        except Exception as e:
            logger.warning(f"[Stage 6a] Error persistiendo en DB: {e}")

    # Señales del mejor experimento → tabla signals
    if output_cfg.get("persistir_signals", False) and resultados:
        try:
            _persistir_signals(resultados, signal_frame, prices)
        except Exception as e:
            logger.warning(f"[Stage 6b] Error persistiendo señales: {e}")


def _persistir_signals(
    resultados: Dict,
    signal_frame: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """Inserta el SignalFrame del mejor experimento (por Sharpe) en tabla signals."""
    from db.database import SessionLocal
    from db.models import Signal

    mejor = max(
        resultados.values(),
        key=lambda r: r.metricas.get("sharpe_ratio", float("-inf")),
    )
    modelo_nombre = mejor.config.nombre
    logger.info(f"[Stage 6b] Mejor experimento: {modelo_nombre}")

    # Tomar la última fila del SignalFrame (señales más recientes)
    ultima_fecha = signal_frame.index.max()
    señales_hoy = signal_frame.loc[ultima_fecha]

    with SessionLocal() as session:
        n = 0
        for ticker, valor in señales_hoy.items():
            if abs(valor) < 0.1:
                continue  # omitir señales muy cercanas a neutral

            tipo = "LONG" if valor > 0 else "SHORT"
            s = Signal(
                ticker=str(ticker),
                fecha=ultima_fecha.date(),
                tipo_signal=tipo,
                valor=float(valor),
                modelo=modelo_nombre,
                confianza=min(abs(float(valor)), 1.0),
                metadata={"sharpe": mejor.metricas.get("sharpe_ratio")},
            )
            session.add(s)
            n += 1
        session.commit()
    logger.info(f"[Stage 6b] {n} señales insertadas en Supabase.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline Fase 2 — research end-to-end",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=str(_ROOT / "scripts" / "config_fase2.yaml"),
        help="Ruta al archivo de configuración YAML",
    )
    parser.add_argument(
        "--experiments", default=None,
        help="Subset de experimentos (ej. '1,2,3'). None → todos los 21.",
    )
    parser.add_argument(
        "--tickers", default=None,
        help="Subset de tickers para testing rápido (ej. 'AAPL,MSFT,GOOG'). "
             "None → universo completo del config.",
    )
    parser.add_argument("--skip-vbt",   action="store_true", help="Omitir Stage 3 (VectorBT).")
    parser.add_argument("--no-mlflow",  action="store_true", help="Desactivar MLflow tracking.")
    parser.add_argument("--no-db",      action="store_true", help="No persistir en Supabase.")
    parser.add_argument("--log-level",  default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    logger.info("=" * 60)
    logger.info("  run_fase2.py — Pipeline Fase 2")
    logger.info("=" * 60)

    # Cargar configuración
    cfg = _load_config(args.config)

    # Overrides por argparse
    if args.skip_vbt:
        cfg.setdefault("vbt_scan", {})["enabled"] = False
    if args.no_mlflow:
        cfg.setdefault("mlflow", {})["enabled"] = False
    if args.no_db:
        cfg.setdefault("output", {})["persistir_db"] = False

    experimentos_subset = None
    if args.experiments:
        try:
            experimentos_subset = [int(x.strip()) for x in args.experiments.split(",")]
        except ValueError:
            logger.error(f"--experiments inválido: '{args.experiments}'. Debe ser ej. '1,2,3'.")
            sys.exit(1)

    wf_cfg       = cfg.get("walk_forward", {})
    mlflow_cfg   = cfg.get("mlflow", {})
    signal_cfg   = cfg.get("signals", {})
    vbt_cfg      = cfg.get("vbt_scan", {})
    backtest_cfg = cfg.get("backtest", {})
    output_cfg   = cfg.get("output", {})

    nombre_exp = mlflow_cfg.get("experiment_name", "walk_forward_factorial")

    # --- Stage 1: Cargar datos ---
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        logger.info(f"Usando subset de tickers por CLI: {tickers}")
    else:
        tickers = _load_universe(wf_cfg.get("universo_file", "universos/sp500.txt"))
    prices, factors = _load_prices_and_factors(
        tickers,
        start=wf_cfg.get("fecha_inicio_oos", "2010-01-01"),
        end=wf_cfg.get("fecha_fin_oos", "2025-01-01"),
    )

    if prices.empty:
        logger.error("[Stage 1] Sin datos de precios. Abortando.")
        sys.exit(1)

    # --- Stage 2: Señales ---
    signal_frame = run_stage2_signals(prices, factors, signal_cfg)

    # --- Stage 3: VectorBT ---
    _vbt_ranking = run_stage3_vbt(prices, signal_frame, vbt_cfg)

    # --- Stage 4: Backtest engine ---
    _bt_result = run_stage4_backtest(prices, signal_frame, backtest_cfg)

    # --- Stage 5: Walk-forward ---
    resultados = run_stage5_walkforward(
        prices=prices,
        factors=factors,
        wf_cfg=wf_cfg,
        mlflow_cfg=mlflow_cfg,
        experimentos_subset=experimentos_subset,
    )

    if not resultados:
        logger.error("[Stage 5] Sin resultados de walk-forward. Abortando persistencia.")
        sys.exit(1)

    # --- Stage 6: Persistencia ---
    run_stage6_persist(resultados, prices, signal_frame, output_cfg, nombre_exp)

    logger.info("=" * 60)
    logger.info("  Pipeline Fase 2 completado.")
    logger.info("=" * 60)

    # Resumen final
    mejor = max(
        resultados.values(),
        key=lambda r: r.metricas.get("sharpe_ratio", float("-inf")),
    )
    def _safe(s):
        try:
            return str(s)
        except Exception:
            return str(s).encode("ascii", "replace").decode("ascii")

    sharpe = mejor.metricas.get("sharpe_ratio", "N/A")
    cagr   = mejor.metricas.get("retorno_anualizado", "N/A")
    mdd    = mejor.metricas.get("max_drawdown", "N/A")
    print(f"\nMejor experimento : {_safe(mejor.config.nombre)}")
    print(f"  Sharpe OOS      : {sharpe:.4f}" if isinstance(sharpe, float) else f"  Sharpe OOS      : {sharpe}")
    print(f"  CAGR OOS        : {cagr:.4f}"   if isinstance(cagr,   float) else f"  CAGR OOS        : {cagr}")
    print(f"  Max Drawdown    : {mdd:.4f}"     if isinstance(mdd,    float) else f"  Max Drawdown    : {mdd}")
    print(f"\nReportes en: {_ROOT / output_cfg.get('directorio_salida', 'data/reportes/fase2')}")
    if mlflow_cfg.get("enabled"):
        print("MLflow UI: mlflow ui --backend-store-uri data/mlruns")


if __name__ == "__main__":
    main()
