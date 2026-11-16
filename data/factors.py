"""
FactorDataManager — descarga FF5+MOM desde Kenneth French y persiste en Supabase.

Envuelve FrenchDataFetcher de ypyquant externamente (ypyquant sigue DB-agnostic).
Gap detection por región: solo descarga si la tabla tiene datos incompletos.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from db.models import FactorData
from ypyquant.data.french_data_fetcher import FrenchDataFetcher

logger = logging.getLogger(__name__)


class FactorDataManager:
    """
    Descarga y persiste factores Fama-French 5 + Momentum en Supabase.

    Uso:
        from db.database import SessionLocal
        from data.factors import FactorDataManager

        with SessionLocal() as session:
            rows = FactorDataManager().fetch_and_save("2020-01-01", "2024-12-31", session)
            print(f"{rows} filas insertadas")
    """

    def __init__(self):
        self._fetcher = FrenchDataFetcher()

    # -----------------------------------------------------------------------
    # API PÚBLICA
    # -----------------------------------------------------------------------

    def fetch_and_save(
        self,
        start_date: str,
        end_date: str,
        session: Session,
        region: str = "USA",
    ) -> int:
        """
        Descarga FF5+MOM para el rango y lo persiste en `factor_data`.

        Gap detection: solo descarga fechas que falten en la DB.

        Args:
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD'
            session:    Sesión SQLAlchemy activa.
            region:     Región para la tabla (default 'USA').

        Returns:
            Número de filas insertadas/actualizadas.
        """
        gaps = self._get_gaps(start_date, end_date, session, region)
        if not gaps:
            logger.info(f"factor_data ({region}): datos completos, nada que insertar")
            return 0

        total = 0
        for gap_start, gap_end in gaps:
            logger.info(f"Descargando factores FF5+MOM {gap_start} → {gap_end}")
            try:
                df = self._fetcher.get_ff5_factors(
                    start_date=gap_start,
                    end_date=gap_end,
                )
                if df is None or df.empty:
                    logger.warning(f"factor_data: FrenchDataFetcher devolvió vacío para {gap_start}→{gap_end}")
                    continue
                total += self._save_to_db(df, session, region)
            except Exception as exc:
                logger.error(f"factor_data: error descargando {gap_start}→{gap_end} — {exc}")

        return total

    def get_rf_rate(
        self,
        date: str,
        session: Session,
        region: str = "USA",
    ) -> Optional[float]:
        """
        Devuelve la tasa RF del día más cercano disponible en factor_data.

        Args:
            date:    'YYYY-MM-DD'
            session: Sesión SQLAlchemy activa.

        Returns:
            RF como float (decimal, ej. 0.0001) o None si no hay datos.
        """
        target = datetime.strptime(date, "%Y-%m-%d").date()

        try:
            # Buscar la fila más cercana a la fecha objetivo (formato largo)
            row = session.execute(
                select(FactorData.fecha, FactorData.valor)
                .where(FactorData.factor_name == "rf")
                .where(FactorData.valor.is_not(None))
                .order_by(func.abs(
                    func.extract("epoch", FactorData.fecha) -
                    func.extract("epoch", target)
                ))
                .limit(1)
            ).first()

            return float(row[1]) if row and row[1] is not None else None

        except Exception as exc:
            logger.warning(f"get_rf_rate: error — {exc}")
            return None

    # -----------------------------------------------------------------------
    # PRIVADO
    # -----------------------------------------------------------------------

    def _get_gaps(
        self,
        start_date: str,
        end_date: str,
        session: Session,
        region: str,
    ):
        """Calcula rangos de fechas sin datos en factor_data."""
        from datetime import timedelta

        desired_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        desired_end = datetime.strptime(end_date, "%Y-%m-%d").date()

        row = session.execute(
            select(func.min(FactorData.fecha), func.max(FactorData.fecha))
            .where(FactorData.factor_name == "mkt_rf")
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

    def _save_to_db(
        self,
        df: pd.DataFrame,
        session: Session,
        region: str,
    ) -> int:
        """
        Persiste el DataFrame de factores en `factor_data` (formato largo).

        FrenchDataFetcher devuelve columnas: MKT/MKT_RF, SMB, HML, RMW, CMA, MOM, RF
        Cada factor se inserta como una fila separada: (factor_name, fecha, valor, source).
        """
        _factor_map = [
            ("mkt_rf", lambda r: r.get("MKT", r.get("MKT_RF"))),
            ("smb",    lambda r: r.get("SMB")),
            ("hml",    lambda r: r.get("HML")),
            ("rmw",    lambda r: r.get("RMW")),
            ("cma",    lambda r: r.get("CMA")),
            ("mom",    lambda r: r.get("MOM")),
            ("rf",     lambda r: r.get("RF")),
        ]
        inserted = 0
        for idx, row in df.iterrows():
            try:
                date_val = idx.date() if hasattr(idx, "date") else idx
                for factor_name, getter in _factor_map:
                    col_val = getter(row)
                    if col_val is None or (hasattr(col_val, '__float__') and not pd.notna(col_val)):
                        continue
                    try:
                        float_val = float(col_val)
                        if not pd.notna(float_val):
                            continue
                    except (TypeError, ValueError):
                        continue

                    session.merge(FactorData(
                        factor_name=factor_name,
                        fecha=date_val,
                        valor=float_val,
                        source="french_data_library",
                    ))
                    inserted += 1
            except Exception as exc:
                logger.error(f"factor_data: error insertando {idx} — {exc}")

        try:
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"factor_data: commit fallido — {exc}")
            return 0

        logger.info(f"factor_data: {inserted} entradas guardadas ({region})")
        return inserted
