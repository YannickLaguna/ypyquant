"""
Adaptador sintético de la fuente de precios: datos deterministas y sin red, para tests y para probar la UI.

Responsabilidad:
- `FuenteSintetica`: paseo geométrico con deriva y volatilidad propias por ticker; la semilla depende solo
  del ticker (hash estable), así que salen siempre los mismos datos. Días hábiles de lunes a viernes desde
  2010-01-04. El benchmark 'SPY' se genera igual que cualquier ticker.
- Deslistados EXPLÍCITOS (`deslistados={'XXX': ('2020-07-03', 'quiebra')}`): la serie se corta en esa fecha y
  `deslistado()` lo declara con su motivo (como lo haría Norgate). `ausentes`: símbolos sin datos (None).
- Precios crudos: `Close` crudo = ajustado * un factor determinista por fecha (>= 1, como una historia con
  dividendos); el ajustado sigue siendo exactamente el de siempre (resultados de los tests intactos).

Se relaciona con:
- ypyquant.data.fuente_precios (interfaz que implementa) y tests/test_fuente_precios_contrato.py (contrato).
- Casi todos los tests de estrategias y de la API (`YPYQUANT_PRECIOS=sintetico` lo activa en `fuente_por_defecto`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping

import numpy as np
import pandas as pd

from ypyquant.data.fuente_precios import Ajuste, Deslistado, FuentePrecios

_FECHA_REFERENCIA_CRUDO = pd.Timestamp("2030-01-01")


class FuenteSintetica(FuentePrecios):
    """Precios sintéticos deterministas; ver la cabecera del módulo."""

    nombre = "sintetica"

    def __init__(self, deriva_anual: float = 0.08, volatilidad_anual: float = 0.22,
                 deslistados: Mapping[str, tuple[str, str | None]] | None = None,
                 ausentes: Collection[str] = ()):
        self.deriva_anual = deriva_anual
        self.volatilidad_anual = volatilidad_anual
        self._deslistados = {s: Deslistado(pd.Timestamp(fecha), motivo) for s, (fecha, motivo) in (deslistados or {}).items()}
        self._ausentes = set(ausentes)

    def ohlcv(self, simbolo: str, fecha_inicio: str, fecha_fin: str, ajuste: Ajuste = "ajustado") -> pd.DataFrame | None:
        if simbolo in self._ausentes:
            return None
        semilla = int(hashlib.sha256(simbolo.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(semilla)
        fechas = pd.bdate_range("2010-01-04", max(pd.Timestamp(fecha_fin), pd.Timestamp("2010-01-05")))
        deriva = self.deriva_anual * (0.5 + rng.random())
        vol = self.volatilidad_anual * (0.6 + 0.8 * rng.random())
        retornos = rng.normal(deriva / 252, vol / np.sqrt(252), len(fechas))
        cierre = 100 * np.exp(np.cumsum(retornos))
        df = pd.DataFrame({
            "Open": cierre * (1 + rng.normal(0, 0.002, len(fechas))),
            "High": cierre * 1.005, "Low": cierre * 0.995, "Close": cierre,
            "Volume": rng.integers(1_000_000, 50_000_000, len(fechas)).astype(float),
            "adjusted_close": cierre,
        }, index=fechas)
        if ajuste == "crudo":
            factor = 1 + 0.1 * np.asarray((_FECHA_REFERENCIA_CRUDO - fechas).days) / 3650
            for columna in ("Open", "High", "Low", "Close"):
                df[columna] = df[columna] * factor
        if simbolo in self._deslistados:
            df = df.loc[: self._deslistados[simbolo].fecha]
        return df.loc[fecha_inicio:fecha_fin]

    def identificador(self, simbolo: str) -> str:
        return f"sint:{simbolo}"

    def deslistado(self, simbolo: str) -> Deslistado | None:
        return self._deslistados.get(simbolo)
