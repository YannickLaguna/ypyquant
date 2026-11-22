"""
_paper_utiles.py — Piezas de apoyo de los tests del paper interno (no es un test).

Responsabilidad:
    - `FuenteManual`: fuente de precios con barras fabricadas a mano (apertura, cierre, huecos, ajustes por
      dividendo) y un registro de las peticiones, para probar el simulador sin red y con casos exactos.
    - `reloj(dia, hora)`: reloj fijo en Nueva York.

Se relaciona con: tests/test_paper_*.py y tests/test_api_paper_interno.py; ypyquant.data.fuente_precios.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ypyquant.data.fuente_precios import FuentePrecios

NY = ZoneInfo("America/New_York")


def reloj(dia: str, hora: str = "20:00:00"):
    """Reloj fijo: `dia` a `hora` (Nueva York). A las 20:00 la sesión de ese día ya está cerrada."""
    return lambda: datetime.fromisoformat(f"{dia}T{hora}").replace(tzinfo=NY)


def barras(inicio: str, fin: str, base: float, deriva: float = 0.002, apertura: float = 0.99) -> pd.DataFrame:
    """Barras diarias deterministas (días hábiles): cierre que sube `deriva` por día y apertura = `apertura` * cierre."""
    fechas = pd.bdate_range(inicio, fin)
    cierre = base * (1 + deriva) ** np.arange(len(fechas)) * (1 + 0.01 * np.sin(np.arange(len(fechas)) * 0.7 + base))
    return pd.DataFrame({"Open": cierre * apertura, "High": cierre * 1.01, "Low": cierre * 0.98, "Close": cierre,
                         "Volume": 1e6, "adjusted_close": cierre}, index=fechas)


class FuenteManual(FuentePrecios):
    """Fuente con DataFrames dados (mutables entre llamadas) y un contador de peticiones por símbolo."""

    nombre = "manual"

    def __init__(self, series: dict[str, pd.DataFrame]):
        self.series = series
        self.peticiones: dict[str, int] = {}

    def ohlcv(self, simbolo, fecha_inicio, fecha_fin, ajuste="ajustado"):
        self.peticiones[simbolo] = self.peticiones.get(simbolo, 0) + 1
        df = self.series.get(simbolo)
        return None if df is None else df.loc[fecha_inicio:fecha_fin]

    def identificador(self, simbolo):
        return f"manual:{simbolo}"

    def deslistado(self, simbolo):
        return None
