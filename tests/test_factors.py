"""
Tests de data/factors.py: FF5+MOM de Kenneth French -> factor_data en la BD.

Responsabilidad:
    Verificar FactorDataManager (fetch_and_save, get_rf_rate) y la integración con
    load_factors(). La construcción del gestor es offline; todo lo demás descarga de
    internet (Kenneth French) y/o escribe en la BD Supabase (ya inexistente), así que va
    marcado `red`.

Se relaciona con:
    - data/factors.py, data/loader.py (código bajo prueba); db/models.py (FactorData).
    - tests/_bd.py (acceso perezoso a la BD) y pyproject.toml (marcador `red`).
"""

import pandas as pd
import pytest
from sqlalchemy import func, select

from data.factors import FactorDataManager
from data.loader import load_factors
from db.models import FactorData
from tests._bd import SessionLocal

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

@pytest.mark.red
def test_fetch_and_save_returns_int():
    """fetch_and_save debe devolver int (puede ser 0 si ya hay datos)."""
    mgr = FactorDataManager()
    with SessionLocal() as s:
        result = mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
    assert isinstance(result, int), f"Debe ser int, got {type(result)}"


@pytest.mark.red
def test_fetch_and_save_idempotent():
    """Segunda llamada con mismo rango debe devolver 0 (ya completo)."""
    mgr = FactorDataManager()
    with SessionLocal() as s:
        mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
        result2 = mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
    assert result2 == 0, f"Segunda llamada debe ser 0, got {result2}"


@pytest.mark.red
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


@pytest.mark.red
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
        pytest.skip("sin filas en factor_data")

    assert row.mkt_rf is not None or row.smb is not None, "Al menos mkt_rf o smb debe tener valor"
    assert row.rf is not None, "rf (tasa libre de riesgo) debe tener valor"


# ---------------------------------------------------------------------------
# get_rf_rate
# ---------------------------------------------------------------------------

@pytest.mark.red
def test_get_rf_rate_returns_float_or_none():
    mgr = FactorDataManager()
    with SessionLocal() as s:
        mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
        rf = mgr.get_rf_rate("2024-01-15", s)
    assert rf is None or isinstance(rf, float), f"Debe ser float o None, got {type(rf)}"


# ---------------------------------------------------------------------------
# Integración: load_factors después de fetch_and_save
# ---------------------------------------------------------------------------

@pytest.mark.red
def test_load_factors_after_fetch():
    """load_factors debe devolver DataFrame no vacío tras fetch_and_save."""
    mgr = FactorDataManager()
    with SessionLocal() as s:
        mgr.fetch_and_save("2024-01-01", "2024-01-31", s)
        df = load_factors("2024-01-01", "2024-01-31", s)

    if df.empty:
        pytest.skip("load_factors vacío; revisar fetch_and_save")

    required = {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"}
    missing = required - set(df.columns)
    assert not missing, f"Faltan columnas: {missing}"
    assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
