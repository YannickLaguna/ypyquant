"""
DataFetcher — descarga OHLCV con triple fallback y persistencia en Supabase.

Fuentes (en orden): yfinance → Twelve Data → Alpha Vantage.
Normaliza al formato estándar: date, open, high, low, close, adjusted_close, volume.
Gap detection: solo descarga rangos de fechas que falten en la DB.

Fallback chain (Fase 4): yfinance → TwelveData → AlphaVantage → IB (Interactive Brokers).
Para añadir IB como 4ª fuente: implementar _try_ib() usando ib_insync.reqHistoricalData()
y añadirla en el bucle de fallbacks de fetch_and_save(). Ver Fase 4 en PRD.md.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Activo

load_dotenv()

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Descarga datos OHLCV con fallback automático y persistencia opcional.

    Uso básico:
        fetcher = DataFetcher()
        df, source = fetcher.fetch("AAPL", "2024-01-01", "2024-12-31")

    Uso con persistencia:
        with SessionLocal() as session:
            rows = fetcher.fetch_and_save("AAPL", "2024-01-01", "2024-12-31", session)
    """

    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.twelve_data_key = os.getenv("TWELVE_DATA_API_KEY", "")

        # Sesión HTTP con User-Agent para yfinance
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

    # -----------------------------------------------------------------------
    # API PÚBLICA
    # -----------------------------------------------------------------------

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Descarga datos OHLCV con triple fallback.

        Returns:
            (DataFrame, source) o (None, None) si todas las fuentes fallan.
            DataFrame columns: date, open, high, low, close, adjusted_close, volume
        """
        logger.info(f"Descargando {ticker} ({start_date} → {end_date})")

        for fn, name in [
            (self._try_yfinance, "yfinance"),
            (self._try_twelvedata, "Twelve Data"),
            (self._try_alphavantage, "Alpha Vantage"),
        ]:
            df, source = fn(ticker, start_date, end_date)
            if df is not None and not df.empty:
                logger.info(f"{ticker}: OK via {name} ({len(df)} registros)")
                return df, source
            logger.warning(f"{ticker}: {name} falló, probando siguiente fuente…")

        logger.error(f"{ticker}: todas las fuentes fallaron")
        return None, None

    def fetch_and_save(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        session: Session,
    ) -> int:
        """
        Fetch + persistencia en tabla `activos`. Solo inserta gaps.

        Returns:
            Número de filas insertadas.
        """
        gaps = self.get_gaps(ticker, start_date, end_date, session)
        if not gaps:
            logger.info(f"{ticker}: datos completos, nada que insertar")
            return 0

        total_inserted = 0
        for gap_start, gap_end in gaps:
            df, source = self.fetch(ticker, gap_start, gap_end)
            if df is None or df.empty:
                continue
            total_inserted += self._save_to_db(df, ticker, source, session)

        return total_inserted

    def fetch_batch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        session: Session,
    ) -> Dict[str, int]:
        """
        Descarga y persiste varios tickers. Retorna {ticker: rows_inserted}.
        """
        results: Dict[str, int] = {}
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{len(tickers)}] {ticker}")
            try:
                results[ticker] = self.fetch_and_save(ticker, start_date, end_date, session)
            except Exception as exc:
                logger.error(f"{ticker}: error inesperado — {exc}")
                results[ticker] = 0
            # Delay conservador entre tickers
            if i < len(tickers):
                time.sleep(1.5)
        return results

    def get_gaps(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        session: Session,
    ) -> List[Tuple[str, str]]:
        """
        Calcula qué rangos de fechas faltan en la DB para este ticker.

        Returns:
            Lista de (gap_start, gap_end) como strings YYYY-MM-DD.
        """
        desired_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        desired_end = datetime.strptime(end_date, "%Y-%m-%d").date()

        row = session.execute(
            select(func.min(Activo.date), func.max(Activo.date))
            .where(Activo.ticker == ticker)
        ).first()

        if row is None or row[0] is None:
            return [(start_date, end_date)]

        existing_start, existing_end = row
        # Normalizar a date (por si Supabase devuelve datetime)
        if hasattr(existing_start, "date"):
            existing_start = existing_start.date()
        if hasattr(existing_end, "date"):
            existing_end = existing_end.date()

        gaps = []
        if desired_start < existing_start:
            gaps.append((
                start_date,
                (existing_start - timedelta(days=1)).strftime("%Y-%m-%d"),
            ))
        if desired_end > existing_end:
            gaps.append((
                (existing_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                end_date,
            ))

        if not gaps:
            logger.debug(f"{ticker}: rango {existing_start} → {existing_end} cubre {start_date} → {end_date}")

        return gaps

    # -----------------------------------------------------------------------
    # FUENTES
    # -----------------------------------------------------------------------

    def _try_yfinance(
        self, ticker: str, start_date: str, end_date: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        try:
            end_inclusive = (
                datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")

            # yf.download() maneja cookie/crumb automáticamente (más robusto que Ticker.history)
            data = yf.download(
                ticker,
                start=start_date,
                end=end_inclusive,
                auto_adjust=False,
                progress=False,
                actions=False,
            )
            time.sleep(0.5)

            if data is None or data.empty:
                return None, None

            return self._normalize_yfinance(data), "yfinance"

        except Exception as exc:
            logger.debug(f"yfinance error ({ticker}): {type(exc).__name__}: {exc}")
            if "429" in str(exc) or "Too Many Requests" in str(exc):
                time.sleep(10)
            return None, None

    def _try_twelvedata(
        self, ticker: str, start_date: str, end_date: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        if not self.twelve_data_key:
            return None, None

        try:
            response = requests.get(
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol": ticker,
                    "interval": "1day",
                    "start_date": start_date,
                    "end_date": end_date,
                    "apikey": self.twelve_data_key,
                    "format": "JSON",
                    "outputsize": 5000,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "error" or "values" not in data:
                logger.debug(f"Twelve Data: respuesta inesperada — {data.get('message', data)}")
                return None, None

            values = data["values"]
            if not values:
                return None, None

            df = pd.DataFrame(values)
            df.columns = ["datetime", "open", "high", "low", "close", "volume"]
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
            df = df.drop("datetime", axis=1)[["date", "open", "high", "low", "close", "volume"]]

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values("date", ascending=True).reset_index(drop=True)

            start_dt = pd.to_datetime(start_date).date()
            end_dt = pd.to_datetime(end_date).date()
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

            if df.empty:
                return None, None

            time.sleep(8)  # 8 req/min free tier → 60/8 ≈ 8s
            return df, "twelvedata"

        except Exception as exc:
            logger.debug(f"Twelve Data error ({ticker}): {type(exc).__name__}")
            return None, None

    def _try_alphavantage(
        self, ticker: str, start_date: str, end_date: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        if not self.alpha_vantage_key:
            return None, None

        try:
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "TIME_SERIES_DAILY",
                    "symbol": ticker,
                    "apikey": self.alpha_vantage_key,
                    "outputsize": "full",
                    "datatype": "json",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            for bad_key in ("Error Message", "Note", "Information"):
                if bad_key in data:
                    logger.debug(f"Alpha Vantage: {bad_key} — {data[bad_key][:80]}")
                    return None, None

            time_series = data.get("Time Series (Daily)")
            if not time_series:
                return None, None

            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)].sort_index()

            if df.empty:
                return None, None

            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"]).dt.date

            time.sleep(12)  # 5 req/min free tier → 60/5 = 12s
            return df, "alphavantage"

        except Exception as exc:
            logger.debug(f"Alpha Vantage error ({ticker}): {type(exc).__name__}")
            return None, None

    # -----------------------------------------------------------------------
    # NORMALIZACIÓN
    # -----------------------------------------------------------------------

    def _normalize_yfinance(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        if "date" not in df.columns and "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)

        if "adj close" in df.columns:
            df.rename(columns={"adj close": "adjusted_close"}, inplace=True)

        required = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]
        available = [c for c in required if c in df.columns]
        df = df[available].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

        return df

    # -----------------------------------------------------------------------
    # PERSISTENCIA
    # -----------------------------------------------------------------------

    def _save_to_db(
        self,
        df: pd.DataFrame,
        ticker: str,
        source: Optional[str],
        session: Session,
    ) -> int:
        inserted = 0
        for _, row in df.iterrows():
            try:
                record = Activo(
                    ticker=ticker,
                    date=row["date"],
                    open=row.get("open"),
                    high=row.get("high"),
                    low=row.get("low"),
                    close=row.get("close"),
                    adjusted_close=row.get("adjusted_close"),
                    volume=int(row["volume"]) if pd.notna(row.get("volume")) else None,
                    source=source,
                )
                session.merge(record)
                inserted += 1
            except Exception as exc:
                logger.error(f"Error insertando {ticker} {row.get('date')}: {exc}")

        session.commit()
        logger.info(f"{ticker}: {inserted} filas guardadas en activos (fuente: {source})")
        return inserted
