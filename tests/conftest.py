"""
Fixtures compartidas y política de marcadores de la batería de tests.

Responsabilidad:
    - Datos sintéticos reproducibles (precios, fechas, generador aleatorio) que
      usan los tests offline, sin red ni base de datos.
    - Raíz del repositorio como fixture, para tests que miran ficheros del repo.
    - Silenciar warnings de pandas que no aportan información en los tests.

Se relaciona con:
    - pyproject.toml: define el marcador `red` y `addopts = "-m 'not red'"`.
    - test_fase2.py, test_ypyquant_api.py: consumen estas fixtures.
    - test_fase1/sp500: tests marcados `red` (necesitan internet); se saltan por defecto.

Cómo ejecutar los tests de red:
    python -m pytest -m red          # solo los de red
    python -m pytest -m "red or not red"   # todo
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ_REPO = Path(__file__).resolve().parent.parent

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
N_DIAS = 504  # ~2 años de sesiones


def pytest_configure(config):
    """Ignora FutureWarning de pandas (pct_change) que ensucian la salida."""
    warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="session")
def raiz_repo() -> Path:
    """Ruta absoluta a la raíz del repositorio."""
    return RAIZ_REPO


@pytest.fixture(scope="session")
def tickers() -> list[str]:
    """Lista de tickers sintéticos usados en los tests de señales."""
    return list(TICKERS)


@pytest.fixture(scope="session")
def fechas() -> pd.DatetimeIndex:
    """Calendario de días hábiles sintético (504 sesiones desde 2022-01-03)."""
    return pd.bdate_range("2022-01-03", periods=N_DIAS)


@pytest.fixture()
def rng() -> np.random.Generator:
    """Generador aleatorio con semilla fija (nuevo para cada test)."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def precios(fechas) -> pd.DataFrame:
    """Precios de cierre sintéticos en formato ancho (índice=fechas, columnas=tickers)."""
    generador = np.random.default_rng(42)
    datos = {}
    for ticker in TICKERS:
        precio_inicial = generador.uniform(50, 300)
        rendimientos = generador.normal(0.0003, 0.015, N_DIAS)
        datos[ticker] = precio_inicial * np.cumprod(1 + rendimientos)
    return pd.DataFrame(datos, index=fechas)
