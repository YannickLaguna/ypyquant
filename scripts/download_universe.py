"""
Script batch para descargar universos de tickers.

Uso:
    python scripts/download_universe.py --tickers AAPL,MSFT,AMZN \\
        --start 2020-01-01 --end 2024-12-31

    python scripts/download_universe.py --file tickers.txt \\
        --start 2020-01-01

    python scripts/download_universe.py --tickers AAPL,MSFT --dry-run

El archivo de tickers puede ser .txt (un ticker por línea) o .csv
(primera columna usada como tickers).
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import SessionLocal
from data.fetcher import DataFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_ticker_file(filepath: str) -> list[str]:
    """Lee tickers desde .txt o .csv."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró: {filepath}")

    tickers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # En CSV usa solo la primera columna
            ticker = line.split(",")[0].strip()
            if ticker:
                tickers.append(ticker.upper())
    return tickers


def main():
    parser = argparse.ArgumentParser(
        description="Descarga batch de datos OHLCV para un universo de tickers."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tickers",
        type=str,
        help="Lista separada por comas: AAPL,MSFT,AMZN",
    )
    source.add_argument(
        "--file",
        type=str,
        help="Archivo .txt o .csv con un ticker por línea",
    )

    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Fecha inicio YYYY-MM-DD (default: 2020-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
        help="Fecha fin YYYY-MM-DD (default: hoy)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Segundos entre tickers (default: 1.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validar lista sin descargar datos",
    )

    args = parser.parse_args()

    # Construir lista de tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = parse_ticker_file(args.file)

    if not tickers:
        logger.error("Lista de tickers vacía")
        sys.exit(1)

    # Validar fechas
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as exc:
        logger.error(f"Formato de fecha inválido: {exc}")
        sys.exit(1)

    logger.info(f"Universo: {len(tickers)} tickers")
    logger.info(f"Rango: {args.start} → {args.end}")

    if args.dry_run:
        logger.info("--- DRY RUN (sin descarga) ---")
        for i, t in enumerate(tickers, 1):
            logger.info(f"  [{i:>3}/{len(tickers)}] {t}")
        logger.info("Dry run completado. No se descargaron datos.")
        return

    fetcher = DataFetcher()
    results = {}
    errors = []

    with SessionLocal() as session:
        for i, ticker in enumerate(tickers, 1):
            eta_tickers = len(tickers) - i
            logger.info(f"[{i}/{len(tickers)}] {ticker} (quedan ~{eta_tickers * args.delay:.0f}s)")

            try:
                rows = fetcher.fetch_and_save(ticker, args.start, args.end, session)
                results[ticker] = rows
                logger.info(f"  → {rows} filas insertadas")
            except Exception as exc:
                logger.error(f"  → ERROR: {exc}")
                errors.append(ticker)
                results[ticker] = 0

            if i < len(tickers):
                time.sleep(args.delay)

    # Resumen
    total_rows = sum(results.values())
    ok = [t for t, r in results.items() if r > 0]
    skipped = [t for t, r in results.items() if r == 0 and t not in errors]

    logger.info("=" * 50)
    logger.info(f"RESUMEN: {len(tickers)} tickers procesados")
    logger.info(f"  Insertados:  {len(ok)} tickers / {total_rows} filas")
    logger.info(f"  Sin cambios: {len(skipped)} tickers (datos ya completos)")
    if errors:
        logger.warning(f"  Errores:     {len(errors)} tickers — {errors}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
