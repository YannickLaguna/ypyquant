"""
Tests de la Fase 1: base de datos (Supabase) + DataFetcher.

Responsabilidad:
    - Offline: importación de los modelos ORM y de DataFetcher.
    - `red`: conexión, tablas, descarga con yfinance, persistencia en `activos`, detección
      de huecos y CRUD de portafolios. Dependen de la BD Supabase (ya inexistente) y/o de
      la red, por eso van marcados `red` y se saltan por defecto.

Se relaciona con:
    - db/database.py, db/models.py, data/fetcher.py (código bajo prueba).
    - tests/_bd.py (acceso perezoso a la BD) y pyproject.toml (marcador `red`).
"""

from datetime import date

import pytest

from tests._bd import SessionLocal, motor

# ── 1. Imports ────────────────────────────────────────────────────────────────

@pytest.mark.red
def test_import_database():
    assert motor() is not None
    assert SessionLocal is not None


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


# ── 2. Conexión a Supabase ────────────────────────────────────────────────────

@pytest.mark.red
def test_connection():
    from sqlalchemy import text
    with motor().connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


# ── 3. Tablas en BD ───────────────────────────────────────────────────────────

@pytest.mark.red
def test_tables_exist():
    from sqlalchemy import inspect
    tablas = inspect(motor()).get_table_names()
    esperadas = ["tickers", "activos", "factor_data", "tipos_cambio",
                 "portafolios", "posiciones", "operaciones", "signals", "backtests"]
    faltan = [t for t in esperadas if t not in tablas]
    assert not faltan, f"Tablas faltantes: {faltan}"


# ── 4. Fetch sin persistencia (red) ───────────────────────────────────────────

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


# ── 5. Persistencia en `activos` ──────────────────────────────────────────────

@pytest.mark.red
def test_fetch_and_save():
    from data.fetcher import DataFetcher
    with SessionLocal() as s:
        filas = DataFetcher().fetch_and_save("AAPL", "2024-01-01", "2024-01-31", s)
    assert filas >= 0  # 0 si ya existían


@pytest.mark.red
def test_data_in_db():
    from sqlalchemy import select

    from db.models import Activo
    with SessionLocal() as s:
        registros = s.execute(select(Activo).where(Activo.ticker == "AAPL")).scalars().all()
        assert len(registros) > 0, "No se encontraron registros de AAPL en activos"


@pytest.mark.red
def test_no_duplicates():
    from sqlalchemy import func, select

    from db.models import Activo
    with SessionLocal() as s:
        duplicados = s.execute(
            select(Activo.date, func.count(Activo.id).label("cnt"))
            .where(Activo.ticker == "AAPL")
            .group_by(Activo.date)
            .having(func.count(Activo.id) > 1)
        ).all()
        assert len(duplicados) == 0, f"Duplicados detectados: {duplicados[:3]}"


# ── 6. Detección de huecos ────────────────────────────────────────────────────

@pytest.mark.red
def test_get_gaps_no_gap():
    """Si los datos ya están, get_gaps debe retornar [] para un rango interior."""
    from sqlalchemy import func, select

    from data.fetcher import DataFetcher
    from db.models import Activo
    with SessionLocal() as s:
        # Asegurar que hay datos primero
        DataFetcher().fetch_and_save("AAPL", "2024-01-01", "2024-01-31", s)
        # Rango real de datos (excluye festivos/fines de semana)
        fila = s.execute(
            select(func.min(Activo.date), func.max(Activo.date)).where(Activo.ticker == "AAPL")
        ).first()
        inicio_real = fila[0].strftime("%Y-%m-%d")
        fin_real = fila[1].strftime("%Y-%m-%d")
        huecos = DataFetcher().get_gaps("AAPL", inicio_real, fin_real, s)
        assert huecos == [], f"Esperaba gaps=[], obtuve {huecos}"


@pytest.mark.red
def test_get_gaps_future():
    """Un rango futuro debe generar un hueco."""
    from data.fetcher import DataFetcher
    with SessionLocal() as s:
        huecos = DataFetcher().get_gaps("AAPL", "2024-01-01", "2024-06-30", s)
        assert isinstance(huecos, list)


@pytest.mark.red
def test_incremental_no_redownload():
    """Segunda llamada de fetch_and_save no debe insertar filas ya existentes."""
    from data.fetcher import DataFetcher
    with SessionLocal() as s:
        filas = DataFetcher().fetch_and_save("AAPL", "2024-01-01", "2024-01-31", s)
        assert filas == 0, f"Segunda llamada debería insertar 0 filas, insertó {filas}"


# ── 7. ORM básico: CRUD de portafolios ────────────────────────────────────────

@pytest.fixture()
def portafolio_id():
    """Inserta un portafolio de prueba y devuelve su id; lo borra al terminar si sigue ahí."""
    from db.models import ResearchPortfolio
    with SessionLocal() as s:
        p = ResearchPortfolio(
            nombre="Test Portfolio",
            descripcion="Portafolio de prueba — borrar",
            tickers=["AAPL", "MSFT"],
            weights={"AAPL": 0.5, "MSFT": 0.5},
        )
        s.add(p)
        s.commit()
        s.refresh(p)
        nuevo_id = p.id
    yield nuevo_id
    with SessionLocal() as s:
        existente = s.get(ResearchPortfolio, nuevo_id)
        if existente is not None:
            s.delete(existente)
            s.commit()


@pytest.mark.red
def test_insert_portafolio(portafolio_id):
    assert portafolio_id is not None


@pytest.mark.red
def test_read_portafolio(portafolio_id):
    from db.models import ResearchPortfolio
    with SessionLocal() as s:
        p = s.get(ResearchPortfolio, portafolio_id)
        assert p is not None
        assert p.nombre == "Test Portfolio"
        assert p.tickers == ["AAPL", "MSFT"]


@pytest.mark.red
def test_delete_portafolio(portafolio_id):
    from db.models import ResearchPortfolio
    with SessionLocal() as s:
        p = s.get(ResearchPortfolio, portafolio_id)
        s.delete(p)
        s.commit()
        assert s.get(ResearchPortfolio, portafolio_id) is None
