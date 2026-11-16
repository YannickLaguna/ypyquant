"""
FREDFetcher — descarga series macroeconómicas de FRED (St. Louis Fed)
y las persiste en la tabla `indicator_data`.

Fuente: https://api.stlouisfed.org/fred/series/observations
API key: variable de entorno FRED_API_KEY

Series predefinidas:
    DFF      — Fed Funds Rate efectivo (diario)
    DGS10    — Rendimiento bono USA 10Y (diario)
    DGS2     — Rendimiento bono USA 2Y  (diario)
    DEXUSEU  — Tipo de cambio EUR/USD oficial Fed (diario)
    CPIAUCSL — CPI USA (mensual)
"""

import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import requests
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from db.models import IndicatorData

load_dotenv()

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

SERIES_CONFIG = {
    "DFF":      {"name": "Fed Funds Rate",        "frequency": "daily"},
    "DGS10":    {"name": "Treasury 10Y",           "frequency": "daily"},
    "DGS2":     {"name": "Treasury 2Y",            "frequency": "daily"},
    "DEXUSEU":  {"name": "EUR/USD (Fed oficial)",  "frequency": "daily"},
    "CPIAUCSL": {"name": "CPI USA",                "frequency": "monthly"},
}


class FREDFetcher:
    """
    Descarga series de FRED y las persiste en `indicator_data`.

    Uso:
        from db.database import SessionLocal
        from data.fred import FREDFetcher

        with SessionLocal() as session:
            rows = FREDFetcher().fetch_and_save(
                ["DFF", "DGS10", "CPIAUCSL"],
                "2018-01-01", "2025-12-31",
                session,
            )
            print(f"{rows} filas insertadas")
    """

    def __init__(self):
        self._api_key = os.getenv("FRED_API_KEY", "")
        if not self._api_key:
            logger.warning("FREDFetcher: FRED_API_KEY no configurada — las descargas fallarán")

    # -----------------------------------------------------------------------
    # API PÚBLICA
    # -----------------------------------------------------------------------

    def fetch_and_save(
        self,
        series_ids: List[str],
        start_date: str,
        end_date: str,
        session: Session,
    ) -> int:
        """
        Descarga series de FRED para el rango y las persiste en indicator_data.

        Gap detection por serie: solo descarga fechas que falten en la DB.

        Args:
            series_ids: Lista de IDs de series FRED (ej. ["DFF", "DGS10"]).
            start_date: Fecha inicio 'YYYY-MM-DD' (inclusive).
            end_date:   Fecha fin 'YYYY-MM-DD' (inclusive).
            session:    Sesión SQLAlchemy activa.

        Returns:
            Número total de filas insertadas/actualizadas.
        """
        total = 0
        for series_id in series_ids:
            try:
                total += self._fetch_series(series_id, start_date, end_date, session)
            except Exception as exc:
                logger.error(f"FRED: error procesando serie {series_id} — {exc}")
        return total

    # -----------------------------------------------------------------------
    # PRIVADO
    # -----------------------------------------------------------------------

    def _fetch_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        session: Session,
    ) -> int:
        """Descarga una serie FRED con gap detection y la persiste."""
        gaps = self._get_gaps(series_id, start_date, end_date, session)
        if not gaps:
            logger.info(f"FRED {series_id}: datos completos, nada que insertar")
            return 0

        inserted = 0
        for gap_start, gap_end in gaps:
            logger.info(f"Descargando FRED {series_id} {gap_start} → {gap_end}")
            observations = self._request_fred(series_id, gap_start, gap_end)
            if observations is None:
                continue
            inserted += self._save_observations(series_id, observations, session)

        return inserted

    def _request_fred(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> Optional[list]:
        """
        Llama a la API de FRED y devuelve la lista de observations.
        Devuelve None si hay error (serie inválida, red caída, etc.).
        """
        if not self._api_key:
            logger.error("FRED: FRED_API_KEY no configurada")
            return None

        params = {
            "series_id":         series_id,
            "observation_start": start_date,
            "observation_end":   end_date,
            "api_key":           self._api_key,
            "file_type":         "json",
        }

        try:
            resp = requests.get(FRED_BASE, params=params, timeout=30)
            if resp.status_code == 400:
                logger.warning(f"FRED: serie inválida o parámetros incorrectos para {series_id} — {resp.text[:200]}")
                return None
            resp.raise_for_status()
            data = resp.json()
            observations = data.get("observations", [])
            logger.debug(f"FRED {series_id}: {len(observations)} observaciones recibidas")
            return observations
        except requests.exceptions.RequestException as exc:
            logger.error(f"FRED: error de red para {series_id} — {exc}")
            return None
        except Exception as exc:
            logger.error(f"FRED: error inesperado para {series_id} — {exc}")
            return None

    def _save_observations(
        self,
        series_id: str,
        observations: list,
        session: Session,
    ) -> int:
        """Persiste observaciones FRED en indicator_data."""
        config = SERIES_CONFIG.get(series_id, {"frequency": "daily"})
        frequency = config["frequency"]
        inserted = 0

        for obs in observations:
            raw_value = obs.get("value", ".")
            # FRED usa "." para valores faltantes
            value = None if raw_value == "." else _safe_float(raw_value)

            date_str = obs.get("date")
            if not date_str:
                continue

            try:
                date_val = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                logger.warning(f"FRED {series_id}: fecha inválida '{date_str}' — ignorando")
                continue

            try:
                session.merge(IndicatorData(
                    date=date_val,
                    entity_id="FRED",
                    metric=series_id,
                    value=value,
                    frequency=frequency,
                    source="fred",
                ))
                inserted += 1
            except Exception as exc:
                logger.error(f"FRED {series_id}: error insertando {date_str} — {exc}")

        try:
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"FRED {series_id}: commit fallido — {exc}")
            return 0

        logger.info(f"FRED {series_id}: {inserted} filas guardadas")
        return inserted

    def _get_gaps(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        session: Session,
    ) -> list:
        """Calcula rangos de fechas sin datos en indicator_data para esta serie."""
        desired_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        desired_end = datetime.strptime(end_date, "%Y-%m-%d").date()

        row = session.execute(
            select(func.min(IndicatorData.date), func.max(IndicatorData.date))
            .where(IndicatorData.entity_id == "FRED")
            .where(IndicatorData.metric == series_id)
        ).first()

        if row is None or row[0] is None:
            return [(start_date, end_date)]

        existing_start, existing_end = row
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

        return gaps


def _safe_float(value) -> Optional[float]:
    """Convierte a float o devuelve None si no es posible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
