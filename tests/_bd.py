"""
Acceso perezoso a la base de datos antigua (Supabase) para los tests marcados `red`.

Responsabilidad:
    Exponer `SessionLocal()` y `motor()` que importan `db.database` solo cuando se
    llaman. Así los módulos de test se pueden importar (y recoger con pytest) sin
    DATABASE_URL ni driver de PostgreSQL; si la BD no está disponible, el test se
    salta con el motivo en lugar de romper la recolección.

Se relaciona con:
    - db/database.py: motor SQLAlchemy real (Supabase, ya inexistente).
    - test_fase1, test_factors, test_loader, test_fred, test_indicators, test_sp500,
      test_fase2 (solo los tests `red`).
    - pyproject.toml: marcador `red`.
"""

import pytest


def _importar_database():
    """Importa db.database; salta el test si falta configuración o driver."""
    try:
        from db import database
    except Exception as error:  # noqa: BLE001 - RuntimeError sin DATABASE_URL, ModuleNotFoundError sin driver...
        pytest.skip(f"BD Supabase no disponible: {type(error).__name__}: {error}")
    return database


def SessionLocal():
    """Devuelve una sesión SQLAlchemy nueva (equivale a db.database.SessionLocal())."""
    return _importar_database().SessionLocal()


def motor():
    """Devuelve el motor SQLAlchemy de db.database."""
    return _importar_database().engine
