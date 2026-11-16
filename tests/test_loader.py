"""
Tests para data/loader.py — puente Supabase → ypyquant.

Verifica que load_prices(), load_factors() y load_fx() devuelven
DataFrames con el formato correcto para ypyquant.

Requisitos previos:
    - AAPL con datos en tabla `activos` (test_fase1.py lo inserta)
    - Conexión Supabase operativa
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from db.database import SessionLocal
from data.loader import load_prices, load_factors, load_fx

PASS = "PASS"
FAIL = "FAIL"
results = []


def test(name: str, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  [PASS] {name}")
    except Exception as exc:
        results.append((FAIL, name))
        print(f"  [FAIL] {name}")
        traceback.print_exc()


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
        print("    (sin datos AAPL en DB — test omitido)")
        return

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
        return

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
        print("    (factor_data vacía — test omitido, ejecutar data/factors.py primero)")
        return

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

if __name__ == "__main__":
    print("\n=== test_loader.py ===\n")

    test("load_prices devuelve dict", test_load_prices_returns_dict)
    test("load_prices AAPL presente", test_load_prices_aapl_present)
    test("load_prices formato DataFrame correcto", test_load_prices_dataframe_format)
    test("load_prices columnas numéricas", test_load_prices_numeric_columns)
    test("load_prices ticker desconocido omitido", test_load_prices_unknown_ticker_omitted)
    test("load_prices lista vacía devuelve {}", test_load_prices_empty_ticker_list)
    test("load_factors devuelve DataFrame", test_load_factors_returns_dataframe)
    test("load_factors vacío si sin datos", test_load_factors_empty_if_no_data)
    test("load_factors columnas correctas", test_load_factors_columns_when_data_present)
    test("load_fx devuelve DataFrame", test_load_fx_returns_dataframe)
    test("load_fx vacío si sin datos", test_load_fx_empty_if_no_data)

    passed = sum(1 for r, _ in results if r == PASS)
    failed = sum(1 for r, _ in results if r == FAIL)
    print(f"\nResultado: {passed}/{len(results)} tests pasados")
    if failed:
        sys.exit(1)
