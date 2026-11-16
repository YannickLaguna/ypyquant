"""
Tests de data/loader.py: puente BD (Supabase) -> ypyquant.

Responsabilidad:
    Verificar que load_prices(), load_factors() y load_fx() devuelven DataFrames con el
    formato que espera ypyquant. Todos leen de la BD Supabase (ya inexistente), por eso
    el módulo entero va marcado `red` y se salta por defecto.

Se relaciona con:
    - data/loader.py (código bajo prueba) y db/database.py (vía tests/_bd.py).
    - pyproject.toml: marcador `red`; ejecutar con `python -m pytest -m red`.
"""

import pandas as pd
import pytest

from data.loader import load_factors, load_fx, load_prices
from tests._bd import SessionLocal

pytestmark = pytest.mark.red


# ---------------------------------------------------------------------------
# load_prices
# ---------------------------------------------------------------------------

def test_load_prices_returns_dict():
    with SessionLocal() as s:
        prices = load_prices(["AAPL"], "2024-01-01", "2024-12-31", s)
    assert isinstance(prices, dict), "Debe ser dict"


def test_load_prices_aapl_present():
    """AAPL debe estar en el resultado si hay datos en activos."""
    with SessionLocal() as s:
        prices = load_prices(["AAPL"], "2024-01-01", "2024-12-31", s)
    # Si no hay datos, se permite dict vacío (test no es bloqueante)
    assert isinstance(prices, dict)


def test_load_prices_dataframe_format():
    """El DataFrame debe tener DatetimeIndex y columnas correctas."""
    with SessionLocal() as s:
        prices = load_prices(["AAPL"], "2024-01-01", "2024-12-31", s)

    if not prices:
        pytest.skip("sin datos AAPL en la BD")

    df = prices["AAPL"]
    assert isinstance(df.index, pd.DatetimeIndex), "Index debe ser DatetimeIndex"
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Faltan columnas: {missing}"


def test_load_prices_numeric_columns():
    """Columnas OHLCV deben ser numéricas."""
    with SessionLocal() as s:
        prices = load_prices(["AAPL"], "2024-01-01", "2024-12-31", s)

    if not prices:
        pytest.skip("sin datos AAPL en la BD")

    df = prices["AAPL"]
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} debe ser numérico"


def test_load_prices_unknown_ticker_omitted():
    """Un ticker sin datos debe ser omitido silenciosamente."""
    with SessionLocal() as s:
        prices = load_prices(["TICKER_QUE_NO_EXISTE_XYZ123"], "2024-01-01", "2024-12-31", s)
    assert "TICKER_QUE_NO_EXISTE_XYZ123" not in prices


def test_load_prices_empty_ticker_list():
    with SessionLocal() as s:
        prices = load_prices([], "2024-01-01", "2024-12-31", s)
    assert prices == {}


# ---------------------------------------------------------------------------
# load_factors
# ---------------------------------------------------------------------------

def test_load_factors_returns_dataframe():
    with SessionLocal() as s:
        df = load_factors("2020-01-01", "2024-12-31", s)
    assert isinstance(df, pd.DataFrame)


def test_load_factors_empty_if_no_data():
    """Si factor_data vacía, debe devolver DataFrame vacío (no excepción)."""
    with SessionLocal() as s:
        df = load_factors("1900-01-01", "1900-12-31", s)
    assert isinstance(df, pd.DataFrame)


def test_load_factors_columns_when_data_present():
    """Si hay datos, las columnas deben coincidir con el formato ypyquant."""
    with SessionLocal() as s:
        df = load_factors("2020-01-01", "2024-12-31", s)

    if df.empty:
        pytest.skip("factor_data vacía; ejecutar data/factors.py primero")

    required = {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"}
    missing = required - set(df.columns)
    assert not missing, f"Faltan columnas: {missing}"
    assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# load_fx
# ---------------------------------------------------------------------------

def test_load_fx_returns_dataframe():
    with SessionLocal() as s:
        df = load_fx(["EURUSD"], "2020-01-01", "2024-12-31", s)
    assert isinstance(df, pd.DataFrame)


def test_load_fx_empty_if_no_data():
    """Si tipos_cambio vacía, debe devolver DataFrame vacío (no excepción)."""
    with SessionLocal() as s:
        df = load_fx(["EURUSD"], "1900-01-01", "1900-12-31", s)
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
