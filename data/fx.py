"""
FXFetcher — descarga tipos de cambio y persiste en tabla `tipos_cambio`.

Pares Fase 1: EURUSD, GBPUSD, JPYUSD, CHFUSD.
Fuente primaria: yfinance (tickers EURUSD=X, etc.)
Fallback: Twelve Data (EUR/USD, etc.)
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from db.models import TipoCambio

logger = logging.getLogger(__name__)

# Pares soportados en Fase 1 y sus tickers yfinance / nombres Twelve Data
_FX_PAIRS: Dict[str, Dict[str, str]] = {
    "EURUSD": {"yf": "EURUSD=X", "td": "EUR/USD"},
    "GBPUSD": {"yf": "GBPUSD=X", "td": "GBP/USD"},
    "JPYUSD": {"yf": "JPYUSD=X", "td": "JPY/USD"},
    "CHFUSD": {"yf": "CHFUSD=X", "td": "CHF/USD"},
}


class FXFetcher:
    """
    Descarga y persiste tipos de cambio diarios.

    Uso:
        from db.database import SessionLocal
        from data.fx import FXFetcher

        with SessionLocal() as session:
            rows = FXFetcher().fetch_and_save(
                ["EURUSD", "GBPUSD"], "2020-01-01", "2024-12-31", session
            )
    """

    def __init__(self):
        self._td_key = os.getenv("TWELVE_DATA_API_KEY", "")
        self._http = requests.Session()
        self._http.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

    # -----------------------------------------------------------------------
    # API PÚBLICA
    # -----------------------------------------------------------------------

    def fetch_and_save(
        self,
        pairs: List[str],
        start_date: str,
        end_date: str,
        session: Session,
    ) -> int:
        """
        Descarga y persiste tipos de cambio en `tipos_cambio`.

        Gap detection por (par, fecha): solo inserta fechas faltantes.

        Args:
            pairs:      Lista de pares a descargar (ej. ['EURUSD', 'GBPUSD']).
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD'
            session:    Sesión SQLAlchemy activa.

        Returns:
            Total de filas insertadas/actualizadas.
        """
        total = 0
        for par in pairs:
            if par not in _FX_PAIRS:
                logger.warning(f"FXFetcher: par '{par}' no soportado. Soportados: {list(_FX_PAIRS)}")
                continue

            gaps = self._get_gaps(par, start_date, end_date, session)
            if not gaps:
                logger.info(f"FX {par}: datos completos, nada que insertar")
                continue

            for gap_start, gap_end in gaps:
                df = self._fetch_pair(par, gap_start, gap_end)
                if df is None or df.empty:
                    logger.warning(f"FX {par}: sin datos para {gap_start}→{gap_end}")
                    continue
                total += self._save_to_db(df, par, session)

        return total

    # -----------------------------------------------------------------------
    # FUENTES
    # -----------------------------------------------------------------------

    def _fetch_pair(
        self, par: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Intenta yfinance, luego Twelve Data."""
        cfg = _FX_PAIRS[par]

        df = self._try_yfinance(cfg["yf"], start_date, end_date)
        if df is not None and not df.empty:
            return df

        logger.warning(f"FX {par}: yfinance falló, probando Twelve Data")
        df = self._try_twelvedata(cfg["td"], start_date, end_date)
        if df is not None and not df.empty:
            return df

        logger.error(f"FX {par}: todas las fuentes fallaron")
        return None

    def _try_yfinance(
        self, yf_ticker: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        try:
            end_inclusive = (
                datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")

            data = yf.download(
                yf_ticker,
                start=start_date,
                end=end_inclusive,
                auto_adjust=False,
                progress=False,
                actions=False,
            )
            time.sleep(0.5)

            if data is None or data.empty:
                return None

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            data = data.reset_index()
            data.columns = [c.lower() for c in data.columns]
            date_col = "date" if "date" in data.columns else "datetime"
            data["fecha"] = pd.to_datetime(data[date_col]).dt.date
            data["tasa"] = pd.to_numeric(data["close"], errors="coerce")

            df = data[["fecha", "tasa"]].dropna().copy()
            df = df[
                (df["fecha"] >= datetime.strptime(start_date, "%Y-%m-%d").date()) &
                (df["fecha"] <= datetime.strptime(end_date, "%Y-%m-%d").date())
            ]
            return df.reset_index(drop=True)

        except Exception as exc:
            logger.debug(f"FX yfinance error ({yf_ticker}): {type(exc).__name__}: {exc}")
            return None

    def _try_twelvedata(
        self, td_symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        if not self._td_key:
            return None
        try:
            response = self._http.get(
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol": td_symbol,
                    "interval": "1day",
                    "start_date": start_date,
                    "end_date": end_date,
                    "apikey": self._td_key,
                    "format": "JSON",
                    "outputsize": 5000,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "error" or "values" not in data:
                logger.debug(f"FX Twelve Data: {data.get('message', data)}")
                return None

            values = data["values"]
            if not values:
                return None

            df = pd.DataFrame(values)
            df["fecha"] = pd.to_datetime(df["datetime"]).dt.date
            df["tasa"] = pd.to_numeric(df["close"], errors="coerce")

            df = df[["fecha", "tasa"]].dropna().sort_values("fecha").reset_index(drop=True)
            time.sleep(8)
            return df

        except Exception as exc:
            logger.debug(f"FX Twelve Data error ({td_symbol}): {type(exc).__name__}")
            return None

    # -----------------------------------------------------------------------
    # GAP DETECTION
    # -----------------------------------------------------------------------

    def _get_gaps(
        self,
        par: str,
        start_date: str,
        end_date: str,
        session: Session,
    ) -> List[Tuple[str, str]]:
        desired_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        desired_end = datetime.strptime(end_date, "%Y-%m-%d").date()

        row = session.execute(
            select(func.min(TipoCambio.fecha), func.max(TipoCambio.fecha))
            .where(TipoCambio.par == par)
        ).first()

        if row is None or row[0] is None:
            return [(start_date, end_date)]

        existing_start, existing_end = row
        if hasattr(existing_start, "date"):
            existing_start = existing_start.date()
        if hasattr(existing_end, "date"):
            existing_end = existing_end.date()

        gaps = []
        from datetime import timedelta
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
        return gaps

    # -----------------------------------------------------------------------
    # PERSISTENCIA
    # -----------------------------------------------------------------------

    def _save_to_db(
        self,
        df: pd.DataFrame,
        par: str,
        session: Session,
    ) -> int:
        inserted = 0
        for _, row in df.iterrows():
            try:
                tasa = float(row["tasa"])
                record = TipoCambio(
                    par=par,
                    fecha=row["fecha"],
                    tasa=tasa,
                    tasa_inversa=(1.0 / tasa) if tasa != 0 else None,
                    source="yfinance",
                )
                session.merge(record)
                inserted += 1
            except Exception as exc:
                logger.error(f"FX {par}: error insertando {row.get('fecha')} — {exc}")

        try:
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"FX {par}: commit fallido — {exc}")
            return 0

        logger.info(f"FX {par}: {inserted} filas guardadas")
        return inserted
