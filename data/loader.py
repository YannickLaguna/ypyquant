"""
Loader — puente Supabase → ypyquant.

Convierte filas ORM de Supabase a DataFrames con el formato esperado por
BacktestEngine, WalkForwardEngine y FamaFrenchModel.

Formato de salida de precios:
    index: DatetimeIndex (Date)
    columns: Open, High, Low, Close, Volume, Adjusted_Close

Formato de salida de factores:
    index: DatetimeIndex
    columns: MKT_RF, SMB, HML, RMW, CMA, MOM, RF
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select

from db.models import Activo, FactorData, TipoCambio, IndicatorData

logger = logging.getLogger(__name__)


def load_prices(
    tickers: List[str],
    start_date: str,
    end_date: str,
    session: Session,
) -> Dict[str, pd.DataFrame]:
    """
    Carga precios OHLCV de la tabla `activos` y devuelve {ticker: DataFrame}.

    El DataFrame de cada ticker tiene:
        index:   DatetimeIndex (Date, frecuencia diaria)
        columns: Open, High, Low, Close, Volume, Adjusted_Close

    Formato compatible con BacktestEngine y WalkForwardEngine de ypyquant.

    Args:
        tickers:    Lista de símbolos a cargar.
        start_date: Fecha inicio 'YYYY-MM-DD' (inclusive).
        end_date:   Fecha fin 'YYYY-MM-DD' (inclusive).
        session:    Sesión SQLAlchemy activa.

    Returns:
        Dict {ticker: DataFrame}. Si un ticker no tiene datos, se omite
        y se emite un warning.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    results: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            rows = session.execute(
                select(Activo)
                .where(Activo.ticker == ticker)
                .where(Activo.date >= start_dt)
                .where(Activo.date <= end_dt)
                .order_by(Activo.date)
            ).scalars().all()

            if not rows:
                logger.warning(f"load_prices: sin datos para {ticker} ({start_date} → {end_date})")
                continue

            df = pd.DataFrame([{
                "Date":           r.date,
                "Open":           r.open,
                "High":           r.high,
                "Low":            r.low,
                "Close":          r.close,
                "Adjusted_Close": r.adjusted_close,
                "Volume":         r.volume,
            } for r in rows])

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()

            results[ticker] = df
            logger.debug(f"load_prices: {ticker} → {len(df)} filas")

        except Exception as exc:
            logger.warning(f"load_prices: error cargando {ticker} — {exc}")

    return results


def load_factors(
    start_date: str,
    end_date: str,
    session: Session,
    region: str = "USA",
) -> pd.DataFrame:
    """
    Carga factores Fama-French + Momentum de la tabla `factor_data` (formato largo).

    Columnas de salida: MKT_RF, SMB, HML, RMW, CMA, MOM, RF
    Valores en decimal (0.01 = 1%).

    Formato compatible con FamaFrenchModel.fit(returns_df, ff_factors_df).

    Returns:
        DataFrame con DatetimeIndex. Vacío si no hay datos en DB.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    try:
        rows = session.execute(
            select(FactorData)
            .where(FactorData.fecha >= start_dt)
            .where(FactorData.fecha <= end_dt)
            .order_by(FactorData.fecha)
        ).scalars().all()

        if not rows:
            logger.warning(f"load_factors: sin datos en factor_data ({start_date} → {end_date})")
            return pd.DataFrame()

        # Construir DataFrame en formato largo y pivotar a formato ancho
        df = pd.DataFrame([{
            "Date":        r.fecha,
            "factor_name": r.factor_name,
            "valor":       r.valor,
        } for r in rows])

        df["Date"] = pd.to_datetime(df["Date"])

        # Pivot: index=Date, columns=factor_name, values=valor
        wide = df.pivot_table(index="Date", columns="factor_name", values="valor", aggfunc="first")

        # Renombrar a nombres esperados por ypyquant
        rename = {
            "mkt_rf": "MKT_RF",
            "smb": "SMB",
            "hml": "HML",
            "rmw": "RMW",
            "cma": "CMA",
            "mom": "MOM",
            "rf": "RF",
        }
        wide = wide.rename(columns=rename).sort_index()
        wide.index.name = "Date"

        logger.debug(f"load_factors: {len(wide)} filas, columnas={list(wide.columns)}")
        return wide

    except Exception as exc:
        logger.warning(f"load_factors: error — {exc}")
        return pd.DataFrame()


def load_fx(
    pairs: List[str],
    start_date: str,
    end_date: str,
    session: Session,
) -> pd.DataFrame:
    """
    Carga tipos de cambio de la tabla `tipos_cambio`.

    Columnas de salida: una columna por par (ej. EURUSD, GBPUSD).
    Index: DatetimeIndex.

    Returns:
        DataFrame pivotado. Vacío si no hay datos.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    try:
        rows = session.execute(
            select(TipoCambio)
            .where(TipoCambio.par.in_(pairs))
            .where(TipoCambio.fecha >= start_dt)
            .where(TipoCambio.fecha <= end_dt)
            .order_by(TipoCambio.fecha)
        ).scalars().all()

        if not rows:
            logger.warning(f"load_fx: sin datos en tipos_cambio para {pairs}")
            return pd.DataFrame()

        df = pd.DataFrame([{
            "fecha": r.fecha,
            "par":   r.par,
            "tasa":  r.tasa,
        } for r in rows])

        df["fecha"] = pd.to_datetime(df["fecha"])
        pivot = df.pivot(index="fecha", columns="par", values="tasa").sort_index()
        pivot.index.name = "Date"

        logger.debug(f"load_fx: {len(pivot)} fechas para {list(pivot.columns)}")
        return pivot

    except Exception as exc:
        logger.warning(f"load_fx: error — {exc}")
        return pd.DataFrame()


def load_indicators(
    entity_ids: List[str],
    metrics: Optional[List[str]],
    start_date: str,
    end_date: str,
    session: Session,
) -> pd.DataFrame:
    """
    Lee indicator_data y devuelve DataFrame pivotado.

    Args:
        entity_ids: Lista de entity_id a cargar (ej. ["FF_USA", "FRED", "AAPL"]).
        metrics:    Lista de métricas a filtrar (ej. ["MKT_RF", "DFF"]). None = todas.
        start_date: 'YYYY-MM-DD' (inclusive).
        end_date:   'YYYY-MM-DD' (inclusive).
        session:    Sesión SQLAlchemy activa.

    Returns:
        DataFrame pivotado con MultiIndex en columnas (entity_id, metric),
        o columna simple si entity_id es único. Index: DatetimeIndex.
        DataFrame vacío si no hay datos (no lanza excepción).
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    try:
        query = (
            select(IndicatorData)
            .where(IndicatorData.entity_id.in_(entity_ids))
            .where(IndicatorData.date >= start_dt)
            .where(IndicatorData.date <= end_dt)
            .order_by(IndicatorData.date)
        )
        if metrics is not None:
            query = query.where(IndicatorData.metric.in_(metrics))

        rows = session.execute(query).scalars().all()

        if not rows:
            logger.warning(
                f"load_indicators: sin datos para {entity_ids} / {metrics} "
                f"({start_date} → {end_date})"
            )
            return pd.DataFrame()

        df = pd.DataFrame([{
            "date":      r.date,
            "entity_id": r.entity_id,
            "metric":    r.metric,
            "value":     r.value,
        } for r in rows])

        df["date"] = pd.to_datetime(df["date"])

        # Pivot: columnas (entity_id, metric) — aplana si entity_id único
        pivot = df.pivot_table(
            index="date", columns=["entity_id", "metric"], values="value", aggfunc="first"
        )
        pivot.index.name = "Date"

        if len(entity_ids) == 1:
            pivot.columns = pivot.columns.droplevel("entity_id")
            pivot.columns.name = None

        logger.debug(f"load_indicators: {len(pivot)} filas, columnas={list(pivot.columns)}")
        return pivot

    except Exception as exc:
        logger.warning(f"load_indicators: error — {exc}")
        return pd.DataFrame()
