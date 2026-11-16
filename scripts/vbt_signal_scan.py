"""
vbt_signal_scan.py — Scan rápido de señales con VectorBT.
=========================================================

Stage 3 del pipeline Fase 2: toma un SignalFrame del stage 2 (SignalCombiner)
y lanza un backtest vectorizado sobre todos los tickers simultáneamente.

VectorBT evalúa 503 tickers en segundos usando NumPy broadcasting.
El output es un ranking de señales/configs por Sharpe bruto (sin costes realistas).
Los top-N configs pasan al stage 4 (backtest_engine.py) para validación precisa.

Uso:
    python scripts/vbt_signal_scan.py \\
        --universe universos/sp500.txt \\
        --start 2020-01-01 \\
        --end   2025-01-01 \\
        [--threshold 0.5] \\
        [--commission 0.001] \\
        [--slippage 0.001] \\
        [--top-n 10]

Requiere:
    pip install "vectorbt>=0.26"
    (ver pyproject.toml [project.optional-dependencies] fast)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core scan function
# ---------------------------------------------------------------------------

def run_vbt_scan(
    prices_df: pd.DataFrame,
    signal_frame: pd.DataFrame,
    commission: float = 0.001,
    slippage: float = 0.001,
    entry_threshold: float = 0.5,
    exit_threshold: float = -0.5,
    freq: str = "1D",
) -> "vbt.Portfolio":
    """
    Ejecuta backtest vectorizado con VectorBT sobre el SignalFrame.

    Args:
        prices_df:       adjusted_close (DatetimeIndex × tickers).
        signal_frame:    SignalFrame ∈ [-1, +1] (mismo shape que prices_df).
        commission:      Comisión por operación (fracción, default 0.1%).
        slippage:        Slippage por operación (fracción, default 0.1%).
        entry_threshold: Umbral para señal de entrada largo (default >0.5).
        exit_threshold:  Umbral para señal de salida (default <-0.5).
        freq:            Frecuencia de datos ('1D' para datos diarios).

    Returns:
        vbt.Portfolio con resultados de todos los tickers.
        Usar pf.stats() para resumen y pf.plot() para gráficos.
    """
    try:
        import vectorbt as vbt
    except ImportError:
        logger.error(
            "VectorBT no instalado. Instala con: pip install 'vectorbt>=0.26'\n"
            "Ver pyproject.toml [project.optional-dependencies] fast"
        )
        raise

    # Alinear signal_frame con prices_df
    common_idx  = prices_df.index.intersection(signal_frame.index)
    common_cols = prices_df.columns.intersection(signal_frame.columns)

    if common_idx.empty or common_cols.empty:
        raise ValueError(
            f"Sin solapamiento entre prices_df y signal_frame. "
            f"Precios: {prices_df.shape}, Señales: {signal_frame.shape}"
        )

    prices  = prices_df.loc[common_idx, common_cols]
    signals = signal_frame.loc[common_idx, common_cols]

    # Generar entradas y salidas booleanas
    entries = signals > entry_threshold   # True donde comprar
    exits   = signals < exit_threshold    # True donde vender/cerrar

    logger.info(
        f"[VBT] Escaneando {len(common_cols)} tickers × {len(common_idx)} días | "
        f"Entradas totales: {entries.sum().sum():,} | Salidas: {exits.sum().sum():,}"
    )

    pf = vbt.Portfolio.from_signals(
        prices,
        entries,
        exits,
        fees=commission,
        slippage=slippage,
        freq=freq,
    )

    logger.info("[VBT] Scan completado.")
    return pf


def scan_and_rank(
    prices_df: pd.DataFrame,
    signal_frame: pd.DataFrame,
    top_n: int = 10,
    commission: float = 0.001,
    slippage: float = 0.001,
    entry_threshold: float = 0.5,
    exit_threshold: float = -0.5,
    sort_by: str = "Sharpe Ratio",
) -> pd.DataFrame:
    """
    Ejecuta el scan y devuelve un ranking de tickers por métrica.

    Args:
        top_n:    Número de configs a devolver en el ranking.
        sort_by:  Columna de pf.stats() por la que ordenar (default: 'Sharpe Ratio').

    Returns:
        DataFrame con las top_n configs ordenadas por sort_by.
    """
    pf = run_vbt_scan(
        prices_df, signal_frame,
        commission=commission, slippage=slippage,
        entry_threshold=entry_threshold, exit_threshold=exit_threshold,
    )

    stats = pf.stats()

    if isinstance(stats, pd.Series):
        # Portfolio agregado (un solo activo) — devolver como DataFrame de 1 fila
        return stats.to_frame().T

    # stats es DataFrame (tickers × métricas)
    if sort_by not in stats.columns:
        available = list(stats.columns)
        logger.warning(f"[VBT] Columna '{sort_by}' no encontrada. Disponibles: {available[:5]}...")
        sort_by = available[0]

    ranking = stats.sort_values(sort_by, ascending=False).head(top_n)
    return ranking


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan rápido de señales con VectorBT (Stage 3 Fase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--universe", default="universos/sp500.txt",
                        help="Archivo con lista de tickers (uno por línea)")
    parser.add_argument("--start", default="2020-01-01", help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--end",   default="2025-01-01", help="Fecha fin YYYY-MM-DD")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Umbral de entrada (señal > threshold → long)")
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--slippage",   type=float, default=0.001)
    parser.add_argument("--top-n",      type=int,   default=10,
                        help="Top N tickers por Sharpe a imprimir")
    parser.add_argument("--output-dir", default="data/reportes/fase2/vbt",
                        help="Directorio donde guardar ranking CSV")
    return parser.parse_args()


def _load_universe(path: str) -> list:
    p = Path(path)
    if not p.exists():
        logger.error(f"Archivo de universo no encontrado: {path}")
        sys.exit(1)
    tickers = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    logger.info(f"[VBT] Universo: {len(tickers)} tickers desde {path}")
    return tickers


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    # --- Verificar VectorBT ---
    try:
        import vectorbt as vbt
        logger.info(f"[VBT] vectorbt {vbt.__version__} disponible.")
    except ImportError:
        logger.error("VectorBT no instalado. Ejecuta: pip install 'vectorbt>=0.26'")
        sys.exit(1)

    # --- Cargar precios desde Supabase/DB ---
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.loader import load_prices
        tickers = _load_universe(args.universe)
        logger.info(f"[VBT] Cargando precios {args.start} → {args.end} ...")
        prices_df = load_prices(tickers, start=args.start, end=args.end)
        prices_df = prices_df.dropna(how="all", axis=1)
        logger.info(f"[VBT] Precios cargados: {prices_df.shape}")
    except Exception as e:
        logger.error(f"Error cargando precios: {e}")
        sys.exit(1)

    # --- Generar señales con TechnicalSignalGenerator ---
    try:
        from ypyquant.signals import TechnicalSignalGenerator
        gen = TechnicalSignalGenerator()
        logger.info("[VBT] Generando señales técnicas ...")
        signal_frame = gen.generate(prices_df)
        logger.info(f"[VBT] SignalFrame shape: {signal_frame.shape}")
    except Exception as e:
        logger.error(f"Error generando señales: {e}")
        sys.exit(1)

    # --- Scan y ranking ---
    try:
        ranking = scan_and_rank(
            prices_df,
            signal_frame,
            top_n=args.top_n,
            commission=args.commission,
            slippage=args.slippage,
            entry_threshold=args.threshold,
            exit_threshold=-args.threshold,
        )
        print(f"\n=== Top {args.top_n} tickers por Sharpe ===")
        print(ranking.to_string())

        # Guardar ranking
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"vbt_ranking_{args.start}_{args.end}.csv"
        ranking.to_csv(out_path)
        logger.info(f"[VBT] Ranking guardado en {out_path}")

    except Exception as e:
        logger.error(f"Error en scan VBT: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
