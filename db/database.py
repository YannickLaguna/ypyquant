"""
Motor SQLAlchemy (sync) + session factory para Sistema_trading.
Conecta a Supabase (PostgreSQL) vía pooler de transacciones.
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL no está definida. "
        "Crea un .env en la raíz del proyecto con la URL de Supabase."
    )

# Normalizar postgres:// → postgresql:// (Supabase a veces usa postgres://)
if DATABASE_URL.startswith("postgres://") and not DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,     # verifica conexión antes de usarla
    pool_recycle=1800,      # recicla conexiones cada 30 min
    connect_args={
        "connect_timeout": 10,
        "options": "-c statement_timeout=30000",  # 30s timeout por query
    },
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

logger.info("Motor de base de datos inicializado (Supabase/PostgreSQL)")
