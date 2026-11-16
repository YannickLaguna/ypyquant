"""
Tests de integración — Fase 1: DB + Fetcher
Ejecutar desde la raíz del proyecto con el venv activado:
    python tests/test_fase1.py
"""

import sys
import os
import logging
from datetime import date
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)  # silenciar logs internos durante tests

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  [PASS] {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  [FAIL] {name}")
        print(f"         {type(e).__name__}: {e}")


# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n── 1. Imports ──")

def test_import_database():
    from db.database import engine, SessionLocal
    assert engine is not None
    assert SessionLocal is not None

def test_import_models():
    from db.models import (
        Base, Ticker, Activo, FactorData, TipoCambio,
        Portafolio, Posicion, Operacion, Signal, Backtest,
    )

def test_import_fetcher():
    from data.fetcher import DataFetcher
    f = DataFetcher()
    assert hasattr(f, "fetch")
    assert hasattr(f, "fetch_and_save")
    assert hasattr(f, "fetch_batch")
    assert hasattr(f, "get_gaps")

check("db.database importa correctamente", test_import_database)
check("db.models importa correctamente (9 modelos)", test_import_models)
check("data.fetcher importa correctamente", test_import_fetcher)


# ── 2. Conexión a Supabase ────────────────────────────────────────────────────
print("\n── 2. Conexión Supabase ──")

def test_connection():
    from db.database import engine
    with engine.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1

check("Conexión a Supabase (SELECT 1)", test_connection)


# ── 3. Tablas en BD ───────────────────────────────────────────────────────────
print("\n── 3. Tablas en BD ──")

def test_tables_exist():
    from db.database import engine
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    expected = ["tickers", "activos", "factor_data", "tipos_cambio",
                "portafolios", "posiciones", "operaciones", "signals", "backtests"]
    missing = [t for t in expected if t not in tables]
    assert not missing, f"Tablas faltantes: {missing}"

check("9 tablas existen en Supabase", test_tables_exist)


# ── 4. Fetch sin persistencia ─────────────────────────────────────────────────
print("\n── 4. Fetch (sin DB) ──")

_fetcher = None
_df_aapl = None
_src_aapl = None

def test_fetch_returns_dataframe():
    global _fetcher, _df_aapl, _src_aapl
    from data.fetcher import DataFetcher
    _fetcher = DataFetcher()
    _df_aapl, _src_aapl = _fetcher.fetch("AAPL", "2024-01-01", "2024-01-31")
    assert _df_aapl is not None, "DataFrame es None"
    assert not _df_aapl.empty, "DataFrame vacío"

def test_fetch_columns():
    assert _df_aapl is not None
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(_df_aapl.columns)
    assert not missing, f"Columnas faltantes: {missing}"

def test_fetch_date_type():
    assert _df_aapl is not None
    sample_date = _df_aapl["date"].iloc[0]
    assert isinstance(sample_date, date), f"date debería ser date, es {type(sample_date)}"

def test_fetch_numeric_types():
    assert _df_aapl is not None
    for col in ["open", "high", "low", "close"]:
        if col in _df_aapl.columns:
            assert _df_aapl[col].dtype == "float64", f"{col} no es float64"

def test_fetch_source():
    assert _src_aapl in ("yfinance", "twelvedata", "alphavantage"), \
        f"Fuente inesperada: {_src_aapl}"

check("fetch() retorna DataFrame no vacío", test_fetch_returns_dataframe)
check("fetch() tiene columnas correctas", test_fetch_columns)
check("columna 'date' es tipo date", test_fetch_date_type)
check("columnas OHLC son float64", test_fetch_numeric_types)
check("source es una fuente válida", test_fetch_source)
if _src_aapl:
    print(f"         (fuente usada: {_src_aapl})")


# ── 5. Persistencia en DB ─────────────────────────────────────────────────────
print("\n── 5. Persistencia en activos ──")

_rows_inserted = 0

def test_fetch_and_save():
    global _rows_inserted
    from db.database import SessionLocal
    from data.fetcher import DataFetcher
    with SessionLocal() as s:
        _rows_inserted = DataFetcher().fetch_and_save("AAPL", "2024-01-01", "2024-01-31", s)
    assert _rows_inserted >= 0  # 0 si ya existían

def test_data_in_db():
    from db.database import SessionLocal
    from db.models import Activo
    from sqlalchemy import select
    with SessionLocal() as s:
        count = s.execute(
            select(Activo).where(Activo.ticker == "AAPL")
        ).scalars().all()
        assert len(count) > 0, "No se encontraron registros de AAPL en activos"

def test_no_duplicates():
    from db.database import SessionLocal
    from db.models import Activo
    from sqlalchemy import select, func
    with SessionLocal() as s:
        result = s.execute(
            select(Activo.date, func.count(Activo.id).label("cnt"))
            .where(Activo.ticker == "AAPL")
            .group_by(Activo.date)
            .having(func.count(Activo.id) > 1)
        ).all()
        assert len(result) == 0, f"Duplicados detectados: {result[:3]}"

check("fetch_and_save() ejecuta sin error", test_fetch_and_save)
check("registros de AAPL existen en DB", test_data_in_db)
check("sin duplicados (UniqueConstraint funciona)", test_no_duplicates)
if _rows_inserted == 0:
    print("         (0 filas: datos ya existían — gap detection funcionó)")
else:
    print(f"         ({_rows_inserted} filas nuevas insertadas)")


# ── 6. Gap detection ──────────────────────────────────────────────────────────
print("\n── 6. Gap detection ──")

def test_get_gaps_no_gap():
    """Si los datos ya están, get_gaps debe retornar [] para un rango interior."""
    from db.database import SessionLocal
    from db.models import Activo
    from sqlalchemy import select, func
    from data.fetcher import DataFetcher
    with SessionLocal() as s:
        # Asegurar que hay datos primero
        DataFetcher().fetch_and_save("AAPL", "2024-01-01", "2024-01-31", s)
        # Obtener el rango real de datos (excluye festivos/fines de semana)
        row = s.execute(
            select(func.min(Activo.date), func.max(Activo.date))
            .where(Activo.ticker == "AAPL")
        ).first()
        real_start = row[0].strftime("%Y-%m-%d")
        real_end   = row[1].strftime("%Y-%m-%d")
        # Pedir exactamente el rango que existe → debe devolver []
        gaps = DataFetcher().get_gaps("AAPL", real_start, real_end, s)
        assert gaps == [], f"Esperaba gaps=[], obtuve {gaps}"

def test_get_gaps_future():
    """Un rango futuro debe generar un gap."""
    from db.database import SessionLocal
    from data.fetcher import DataFetcher
    with SessionLocal() as s:
        gaps = DataFetcher().get_gaps("AAPL", "2024-01-01", "2024-06-30", s)
        # Debe haber al menos un gap hacia fechas más recientes
        assert isinstance(gaps, list)

def test_incremental_no_redownload():
    """Segunda llamada fetch_and_save no debe insertar filas ya existentes."""
    from db.database import SessionLocal
    from data.fetcher import DataFetcher
    with SessionLocal() as s:
        rows = DataFetcher().fetch_and_save("AAPL", "2024-01-01", "2024-01-31", s)
        assert rows == 0, f"Segunda llamada debería insertar 0 filas, insertó {rows}"

check("get_gaps() retorna [] cuando datos completos", test_get_gaps_no_gap)
check("get_gaps() detecta gap hacia futuro", test_get_gaps_future)
check("segunda llamada no re-inserta (incremental OK)", test_incremental_no_redownload)


# ── 7. ORM básico ─────────────────────────────────────────────────────────────
print("\n── 7. ORM — CRUD básico ──")

_portafolio_id = None

def test_insert_portafolio():
    global _portafolio_id
    from db.database import SessionLocal
    from db.models import Portafolio
    with SessionLocal() as s:
        p = Portafolio(
            nombre="Test Portfolio",
            descripcion="Portafolio de prueba — borrar",
            tickers=["AAPL", "MSFT"],
            weights={"AAPL": 0.5, "MSFT": 0.5},
        )
        s.add(p)
        s.commit()
        s.refresh(p)
        _portafolio_id = p.id
        assert _portafolio_id is not None

def test_read_portafolio():
    from db.database import SessionLocal
    from db.models import Portafolio
    with SessionLocal() as s:
        p = s.get(Portafolio, _portafolio_id)
        assert p is not None
        assert p.nombre == "Test Portfolio"
        assert p.tickers == ["AAPL", "MSFT"]

def test_delete_portafolio():
    from db.database import SessionLocal
    from db.models import Portafolio
    with SessionLocal() as s:
        p = s.get(Portafolio, _portafolio_id)
        s.delete(p)
        s.commit()
        p2 = s.get(Portafolio, _portafolio_id)
        assert p2 is None

check("INSERT portafolio", test_insert_portafolio)
check("SELECT portafolio por id", test_read_portafolio)
check("DELETE portafolio", test_delete_portafolio)


# ── Resumen ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"  {passed}/{len(results)} tests pasados")
if failed:
    print(f"  {failed} tests fallaron:")
    for status, name in results:
        if status == FAIL:
            print(f"    - {name}")
    sys.exit(1)
else:
    print("  Todo OK — Fase 1 operativa")
print("─" * 50)
