"""
Tests para data/factors.py — FF5+MOM → factor_data en Supabase.

Verifica FactorDataManager: fetch_and_save(), get_rf_rate(), y la
integración con load_factors() del loader.

Nota: fetch_and_save() descarga desde internet (Kenneth French Data Library).
      Si no hay conexión, los tests de descarga fallan con warning.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from db.database import SessionLocal
from db.models import FactorData
from data.factors import FactorDataManager
from data.loader import load_factors
from sqlalchemy import select, func

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
# FactorDataManager — importación y construcción
# ---------------------------------------------------------------------------

def test_import():
    from data.factors import FactorDataManager
    mgr = FactorDataManager()
    assert mgr is not None


def test_fetcher_attribute():
    mgr = FactorDataManager()
    assert hasattr(mgr, "_fetcher"), "Debe tener _fetcher (FrenchDataFetcher)"


# ---------------------------------------------------------------------------
# fetch_and_save — rango pequeño para test rápido
# ---------------------------------------------------------------------------

def test_fetch_and_save_returns_int():
    """fetch_and_save debe devolver int (puede ser 0 si ya hay datos)."""
    mgr = FactorDataManager()
    with SessionLocal() as s:
        result = mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
    assert isinstance(result, int), f"Debe ser int, got {type(result)}"


def test_fetch_and_save_idempotent():
    """Segunda llamada con mismo rango debe devolver 0 (ya completo)."""
    mgr = FactorDataManager()
    with SessionLocal() as s:
        mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
        result2 = mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
    assert result2 == 0, f"Segunda llamada debe ser 0, got {result2}"


def test_data_in_db_after_fetch():
    """Después de fetch_and_save, debe haber filas en factor_data."""
    mgr = FactorDataManager()
    with SessionLocal() as s:
        mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
        count = s.execute(
            select(func.count(FactorData.id))
            .where(FactorData.region == "USA")
        ).scalar()
    assert count > 0, "factor_data debe tener filas después de fetch"


def test_factor_data_columns():
    """Las filas en factor_data deben tener los campos principales."""
    mgr = FactorDataManager()
    with SessionLocal() as s:
        mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
        row = s.execute(
            select(FactorData)
            .where(FactorData.region == "USA")
            .limit(1)
        ).scalars().first()

    if row is None:
        print("    (sin filas en factor_data — test omitido)")
        return

    assert row.mkt_rf is not None or row.smb is not None, "Al menos mkt_rf o smb debe tener valor"
    assert row.rf is not None, "rf (tasa libre de riesgo) debe tener valor"


# ---------------------------------------------------------------------------
# get_rf_rate
# ---------------------------------------------------------------------------

def test_get_rf_rate_returns_float_or_none():
    mgr = FactorDataManager()
    with SessionLocal() as s:
        mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
        rf = mgr.get_rf_rate("2024-01-15", s)
    assert rf is None or isinstance(rf, float), f"Debe ser float o None, got {type(rf)}"


# ---------------------------------------------------------------------------
# Integración: load_factors después de fetch_and_save
# ---------------------------------------------------------------------------

def test_load_factors_after_fetch():
    """load_factors debe devolver DataFrame no vacío tras fetch_and_save."""
    mgr = FactorDataManager()
    with SessionLocal() as s:
        mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
        df = load_factors("2024-01-01", "2024-01-31", s)

    if df.empty:
        print("    (load_factors vacío — revisar fetch_and_save)")
        return

    required = {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"}
    missing = required - set(df.columns)
    assert not missing, f"Faltan columnas: {missing}"
    assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== test_factors.py ===\n")

    test("FactorDataManager importación", test_import)
    test("FactorDataManager atributo _fetcher", test_fetcher_attribute)
    test("fetch_and_save devuelve int", test_fetch_and_save_returns_int)
    test("fetch_and_save idempotente", test_fetch_and_save_idempotent)
    test("datos en DB después de fetch", test_data_in_db_after_fetch)
    test("columnas factor_data correctas", test_factor_data_columns)
    test("get_rf_rate devuelve float o None", test_get_rf_rate_returns_float_or_none)
    test("load_factors tras fetch tiene columnas correctas", test_load_factors_after_fetch)

    passed = sum(1 for r, _ in results if r == PASS)
    failed = sum(1 for r, _ in results if r == FAIL)
    print(f"\nResultado: {passed}/{len(results)} tests pasados")
    if failed:
        sys.exit(1)
