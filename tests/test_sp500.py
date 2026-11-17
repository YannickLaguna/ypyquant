"""
Tests de scripts/fetch_sp500.py (Fase 1): lista de tickers del S&P 500 desde Wikipedia.

Responsabilidad:
    - Offline: el script es importable y expone sus funciones públicas.
    - `red`: fetch_sp500_tickers() y fetch_sp500_wiki_df() descargan de Wikipedia
      (conversión BRK.B -> BRK-B, columnas esperadas).

Se relaciona con:
    - scripts/fetch_sp500.py (código bajo prueba).
    - pyproject.toml (marcador `red`).
"""

import pytest

COLUMNAS_ESPERADAS = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]


# ── 1. Import ─────────────────────────────────────────────────────────────────


def test_importable():
    import scripts.fetch_sp500  # noqa: F401


def test_functions_exist():
    from scripts.fetch_sp500 import fetch_sp500_tickers, fetch_sp500_wiki_df, upsert_tickers_metadata
    assert callable(fetch_sp500_tickers)
    assert callable(fetch_sp500_wiki_df)
    assert callable(upsert_tickers_metadata)


# ── 2. fetch_sp500_tickers() — red ────────────────────────────────────────────


@pytest.mark.red
def test_returns_list():
    from scripts.fetch_sp500 import fetch_sp500_tickers
    tickers = fetch_sp500_tickers()
    assert isinstance(tickers, list), "No es lista"
    assert len(tickers) > 400, f"Muy pocos tickers: {len(tickers)}"
    assert all(isinstance(t, str) for t in tickers), "No todos son strings"


@pytest.mark.red
def test_dot_to_dash():
    from scripts.fetch_sp500 import fetch_sp500_tickers
    tickers = fetch_sp500_tickers()
    # BRK.B debe haberse convertido a BRK-B
    assert "BRK-B" in tickers, f"BRK-B no encontrado (¿aún BRK.B?). Muestra: {tickers[:10]}"
    assert "BRK.B" not in tickers, "BRK.B no fue convertido a BRK-B"


# ── 3. fetch_sp500_wiki_df() — columnas ───────────────────────────────────────



@pytest.mark.red
def test_df_columns():
    from scripts.fetch_sp500 import fetch_sp500_wiki_df
    df = fetch_sp500_wiki_df()
    missing = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    assert not missing, f"Columnas faltantes: {missing}. Disponibles: {list(df.columns)}"


@pytest.mark.red
def test_df_no_dots_in_symbol():
    from scripts.fetch_sp500 import fetch_sp500_wiki_df
    df = fetch_sp500_wiki_df()
    has_dot = df["Symbol"].str.contains(r"\.", regex=True).any()
    assert not has_dot, "Símbolos con punto encontrados (esperado: ya convertidos a guión)"


# ── 4. DB integration (opcional, requiere --db) ───────────────────────────────
