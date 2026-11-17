"""
Tests de data/factors.py: construcción de FactorDataManager (FF5+MOM de Kenneth French).

Responsabilidad:
    Verificar offline la construcción de FactorDataManager. Los tests de
    fetch_and_save/get_rf_rate/load_factors escribían en la BD Supabase (ya inexistente)
    y se eliminaron.

Se relaciona con:
    - data/factors.py (código bajo prueba).
"""

from data.factors import FactorDataManager

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
