"""
Tests para scripts/fetch_sp500.py — Fase 1
Ejecutar desde la raíz del proyecto:
    python -X utf8 tests/test_sp500.py

Tests offline (sin red):
    1. Script importable
    2. fetch_sp500_tickers() devuelve lista de strings
    3. BRK.B → BRK-B (dot-to-dash)
    4. fetch_sp500_wiki_df() devuelve DataFrame con columnas esperadas

Test de integración con BD (requiere --db):
    5. Tabla tickers contiene >=1 registro con sector no nulo (tras --metadata)
"""

import sys
import os
import io
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PASS = "PASS"
FAIL = "FAIL"
results = []

RUN_DB = "--db" in sys.argv


def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  [PASS] {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  [FAIL] {name}")
        print(f"         {type(e).__name__}: {e}")


# ── 1. Import ─────────────────────────────────────────────────────────────────
print("\n── 1. Import ──")


def _test_importable():
    import scripts.fetch_sp500  # noqa: F401


def _test_functions_exist():
    from scripts.fetch_sp500 import fetch_sp500_tickers, fetch_sp500_wiki_df, upsert_tickers_metadata
    assert callable(fetch_sp500_tickers)
    assert callable(fetch_sp500_wiki_df)
    assert callable(upsert_tickers_metadata)


check("script importable", _test_importable)
check("funciones públicas existen", _test_functions_exist)


# ── 2. fetch_sp500_tickers() — red ────────────────────────────────────────────
print("\n── 2. fetch_sp500_tickers() ──")


def _test_returns_list():
    from scripts.fetch_sp500 import fetch_sp500_tickers
    tickers = fetch_sp500_tickers()
    assert isinstance(tickers, list), "No es lista"
    assert len(tickers) > 400, f"Muy pocos tickers: {len(tickers)}"
    assert all(isinstance(t, str) for t in tickers), "No todos son strings"


def _test_dot_to_dash():
    from scripts.fetch_sp500 import fetch_sp500_tickers
    tickers = fetch_sp500_tickers()
    # BRK.B debe haberse convertido a BRK-B
    assert "BRK-B" in tickers, f"BRK-B no encontrado (¿aún BRK.B?). Muestra: {tickers[:10]}"
    assert "BRK.B" not in tickers, "BRK.B no fue convertido a BRK-B"


check("devuelve lista de strings >=400", _test_returns_list)
check("BRK.B → BRK-B (dot-to-dash)", _test_dot_to_dash)


# ── 3. fetch_sp500_wiki_df() — columnas ───────────────────────────────────────
print("\n── 3. fetch_sp500_wiki_df() ──")

EXPECTED_COLS = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]


def _test_df_columns():
    from scripts.fetch_sp500 import fetch_sp500_wiki_df
    df = fetch_sp500_wiki_df()
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    assert not missing, f"Columnas faltantes: {missing}. Disponibles: {list(df.columns)}"


def _test_df_no_dots_in_symbol():
    from scripts.fetch_sp500 import fetch_sp500_wiki_df
    df = fetch_sp500_wiki_df()
    has_dot = df["Symbol"].str.contains(r"\.", regex=True).any()
    assert not has_dot, "Símbolos con punto encontrados (esperado: ya convertidos a guión)"


check("DataFrame tiene columnas esperadas", _test_df_columns)
check("Símbolos sin puntos en DataFrame", _test_df_no_dots_in_symbol)


# ── 4. DB integration (opcional, requiere --db) ───────────────────────────────
if RUN_DB:
    print("\n── 4. DB integration ──")

    def _test_tickers_with_sector():
        from db.database import SessionLocal
        from db.models import Ticker
        with SessionLocal() as s:
            n = s.query(Ticker).filter(Ticker.sector.isnot(None)).count()
            assert n >= 1, f"0 tickers con sector en BD (ejecuta --metadata primero)"

    def _test_aapl_metadata():
        from db.database import SessionLocal
        from db.models import Ticker
        with SessionLocal() as s:
            t = s.query(Ticker).filter_by(symbol="AAPL").first()
            assert t is not None, "AAPL no encontrado en tabla tickers"
            assert t.sector is not None, "AAPL.sector es None"
            assert t.name is not None, "AAPL.name es None"
            assert t.country == "USA", f"AAPL.country={t.country}"
            assert t.asset_type == "equity", f"AAPL.asset_type={t.asset_type}"

    check(">=1 ticker con sector no nulo", _test_tickers_with_sector)
    check("AAPL tiene metadatos completos", _test_aapl_metadata)
else:
    print("\n── 4. DB integration — SKIPPED (usa --db para ejecutar) ──")


# ── Resumen ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
passed = sum(1 for s, _ in results if s == PASS)
failed = sum(1 for s, _ in results if s == FAIL)
print(f"Resultado: {passed}/{len(results)} tests pasaron")
if failed:
    print(f"FALLARON {failed} tests")
    sys.exit(1)
else:
    print("Todos los tests pasaron.")
