"""
Tests de la Fase 1: DataFetcher.

Responsabilidad:
    - Offline: importación de los modelos ORM y de DataFetcher.
    - `red`: descarga con yfinance (fetch sin persistencia); se saltan por defecto.

Se relaciona con:
    - db/database.py, db/models.py, data/fetcher.py (código bajo prueba).
    - pyproject.toml (marcador `red`).
"""

from datetime import date

import pytest

# ── 1. Imports ────────────────────────────────────────────────────────────────

def test_import_models():
    from db.models import (  # noqa: F401
        Activo,
        Backtest,
        Base,
        FactorData,
        Operacion,
        Posicion,
        ResearchPortfolio,
        Signal,
        Ticker,
        TipoCambio,
    )


def test_import_fetcher():
    from data.fetcher import DataFetcher
    f = DataFetcher()
    assert hasattr(f, "fetch")
    assert hasattr(f, "fetch_and_save")
    assert hasattr(f, "fetch_batch")
    assert hasattr(f, "get_gaps")


# ── 2. Fetch sin persistencia (red) ───────────────────────────────────────────

@pytest.fixture(scope="module")
def descarga_aapl():
    """Descarga AAPL de enero de 2024 una sola vez; devuelve (DataFrame, fuente)."""
    from data.fetcher import DataFetcher
    return DataFetcher().fetch("AAPL", "2024-01-01", "2024-01-31")


@pytest.mark.red
def test_fetch_returns_dataframe(descarga_aapl):
    df, _ = descarga_aapl
    assert df is not None, "DataFrame es None"
    assert not df.empty, "DataFrame vacío"


@pytest.mark.red
def test_fetch_columns(descarga_aapl):
    df, _ = descarga_aapl
    faltan = {"date", "open", "high", "low", "close", "volume"} - set(df.columns)
    assert not faltan, f"Columnas faltantes: {faltan}"


@pytest.mark.red
def test_fetch_date_type(descarga_aapl):
    df, _ = descarga_aapl
    primera = df["date"].iloc[0]
    assert isinstance(primera, date), f"date debería ser date, es {type(primera)}"


@pytest.mark.red
def test_fetch_numeric_types(descarga_aapl):
    df, _ = descarga_aapl
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            assert df[col].dtype == "float64", f"{col} no es float64"


@pytest.mark.red
def test_fetch_source(descarga_aapl):
    _, fuente = descarga_aapl
    assert fuente in ("yfinance", "twelvedata", "alphavantage"), f"Fuente inesperada: {fuente}"
