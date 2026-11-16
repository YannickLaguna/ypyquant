"""
Elimina y recrea las tablas de Fase 1 en Supabase.
CUIDADO: borra todos los datos existentes en estas tablas.

Uso:
    python scripts/reset_tables.py
    python scripts/reset_tables.py --confirm   # sin prompt interactivo
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text, inspect
from db.database import engine
from db.models import Base

TABLES = [
    "signals", "backtests", "operaciones", "posiciones",
    "portafolios", "tipos_cambio", "factor_data", "activos", "tickers",
]


def main():
    auto_confirm = "--confirm" in sys.argv

    if not auto_confirm:
        print("Este script elimina y recrea las siguientes tablas en Supabase:")
        for t in TABLES:
            print(f"  - {t}")
        confirm = input("\nEscribe 'si' para continuar: ").strip().lower()
        if confirm != "si":
            print("Cancelado.")
            sys.exit(0)

    with engine.begin() as conn:
        # Desactivar FK checks temporalmente para poder borrar en cualquier orden
        conn.execute(text("SET session_replication_role = replica"))
        for table in TABLES:
            inspector = inspect(engine)
            if table in inspector.get_table_names():
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                print(f"  Eliminada: {table}")
            else:
                print(f"  No existia: {table}")
        conn.execute(text("SET session_replication_role = DEFAULT"))

    print("\nRecreando tablas con schema actualizado...")
    Base.metadata.create_all(engine)
    print("Listo. Tablas recreadas:")
    inspector = inspect(engine)
    for t in inspector.get_table_names():
        if t in TABLES:
            cols = [c["name"] for c in inspector.get_columns(t)]
            print(f"  {t}: {cols}")


if __name__ == "__main__":
    main()
