"""
fetch_sp500.py — Obtiene el universo S&P 500 desde Wikipedia y opcionalmente
                 descarga datos OHLCV para todos los componentes.

Uso:
    python scripts/fetch_sp500.py                        # genera universos/sp500.txt
    python scripts/fetch_sp500.py --metadata             # upsert metadatos en tabla tickers
    python scripts/fetch_sp500.py --download             # descarga OHLCV (largo)
    python scripts/fetch_sp500.py --download --start 2018-01-01 --end 2024-12-31

Nota BRK.B / BF.B: Wikipedia usa punto, yfinance usa guión. La conversión
es automática (BRK.B → BRK-B).

SURVIVORSHIP BIAS WARNING:
    La lista generada refleja los constituyentes ACTUALES del S&P 500.
    Usar esta lista para backtesting introduce sesgo de supervivencia: los tickers
    eliminados por quiebra, fusión o exclusión no aparecen.
    Para backtesting riguroso con universo point-in-time ver: Fase 2 / CRSP / Norgate.
"""

import argparse
import io
import logging
import os
import sys

import pandas as pd
import requests

# Añadir raíz del proyecto al path para importar db / data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "universos", "sp500.txt"
)


def fetch_sp500_wiki_df() -> "pd.DataFrame":
    """
    Scrape Wikipedia y devuelve el DataFrame completo con columnas:
    Symbol, Security, GICS Sector, GICS Sub-Industry, …

    Los símbolos en la columna Symbol ya están normalizados (punto → guión).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    logger.info("Descargando lista S&P 500 desde Wikipedia…")
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]  # primera tabla = componentes actuales
    df["Symbol"] = df["Symbol"].str.strip().str.replace(".", "-", regex=False)
    return df


def fetch_sp500_tickers() -> list[str]:
    """
    Scrape Wikipedia para obtener los tickers del S&P 500.

    Convierte puntos a guiones para compatibilidad con yfinance:
        BRK.B → BRK-B
        BF.B  → BF-B
    """
    df = fetch_sp500_wiki_df()
    tickers = df["Symbol"].tolist()
    logger.info(f"S&P 500: {len(tickers)} tickers obtenidos")
    return tickers


def save_universe(tickers: list[str], output_path: str) -> None:
    """Guarda tickers en un archivo de texto, uno por línea."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ticker in tickers:
            f.write(ticker + "\n")
    logger.info(f"Universo guardado en: {output_path}")


def upsert_tickers_metadata(df_wiki: "pd.DataFrame", session) -> int:
    """
    Upsert metadatos de los constituyentes S&P 500 en la tabla `tickers`.

    Mapeo desde Wikipedia:
        Symbol          → symbol
        Security        → name
        GICS Sector     → sector
        GICS Sub-Industry → industry
    Constantes: country="USA", currency="USD", asset_type="equity",
                index_memberships={"SP500": True}

    Devuelve el número de registros procesados.
    """
    from db.models import Ticker

    # Nombres de columna normalizados (Wikipedia puede variar levemente)
    col_map = {
        "Symbol": "symbol",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "industry",
    }
    # Verificar que las columnas esperadas existen
    missing = [c for c in col_map if c not in df_wiki.columns]
    if missing:
        raise ValueError(f"Columnas no encontradas en tabla Wikipedia: {missing}. "
                         f"Disponibles: {list(df_wiki.columns)}")

    count = 0
    for _, row in df_wiki.iterrows():
        symbol = row["Symbol"]
        existing = session.query(Ticker).filter_by(symbol=symbol).first()
        if existing is None:
            ticker_obj = Ticker(
                symbol=symbol,
                name=row.get("Security"),
                sector=row.get("GICS Sector"),
                industry=row.get("GICS Sub-Industry"),
                country="USA",
                currency="USD",
                asset_type="equity",
                index_memberships={"SP500": True},
            )
            session.add(ticker_obj)
        else:
            existing.name = row.get("Security") or existing.name
            existing.sector = row.get("GICS Sector") or existing.sector
            existing.industry = row.get("GICS Sub-Industry") or existing.industry
            existing.country = existing.country or "USA"
            existing.currency = existing.currency or "USD"
            existing.asset_type = existing.asset_type or "equity"
            memberships = existing.index_memberships or {}
            memberships["SP500"] = True
            existing.index_memberships = memberships
        count += 1

    session.commit()
    logger.info(f"Upsert completado: {count} tickers en tabla `tickers`")
    return count


def download_ohlcv(tickers: list[str], start_date: str, end_date: str) -> None:
    """
    Descarga datos OHLCV para todos los tickers del S&P 500.
    Llama a DataFetcher para cada ticker con persistencia en Supabase.
    """
    from db.database import SessionLocal
    from data.fetcher import DataFetcher

    fetcher = DataFetcher()
    total = 0
    errors = 0

    with SessionLocal() as session:
        for i, ticker in enumerate(tickers, 1):
            try:
                rows = fetcher.fetch_and_save(ticker, start_date, end_date, session)
                total += rows
                if i % 50 == 0:
                    logger.info(f"Progreso: {i}/{len(tickers)} tickers procesados ({total} filas totales)")
            except Exception as exc:
                logger.warning(f"Error en {ticker}: {exc}")
                errors += 1

    logger.info(f"Descarga completada: {len(tickers) - errors}/{len(tickers)} tickers OK, {total} filas insertadas")


def main():
    parser = argparse.ArgumentParser(description="Fetch S&P 500 universe from Wikipedia")
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Upsert metadatos (name, sector, industry) en tabla tickers",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Descargar datos OHLCV para todos los tickers",
    )
    parser.add_argument(
        "--start",
        default="2018-01-01",
        help="Fecha inicio para descarga OHLCV (default: 2018-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2025-12-31",
        help="Fecha fin para descarga OHLCV (default: 2025-12-31)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help=f"Ruta del archivo de salida (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    df_wiki = fetch_sp500_wiki_df()
    tickers = df_wiki["Symbol"].tolist()
    save_universe(tickers, args.output)
    print(f"\n{len(tickers)} tickers guardados en {args.output}")

    if args.metadata:
        from db.database import SessionLocal
        logger.info("Upserting metadatos en tabla tickers…")
        with SessionLocal() as session:
            n = upsert_tickers_metadata(df_wiki, session)
        print(f"{n} tickers con metadatos actualizados en la BD")

    if args.download:
        logger.info(f"Iniciando descarga OHLCV para {len(tickers)} tickers ({args.start} → {args.end})")
        download_ohlcv(tickers, args.start, args.end)


if __name__ == "__main__":
    main()
