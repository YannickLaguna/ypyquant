"""
Pesos de mercado w_mkt para Black-Litterman (prior de equilibrio).

Responsabilidad:
- `calcular_w_mkt_volumen`: proxy de w_mkt por $-volumen medio móvil, walk-forward (sin
  look-ahead), porte del TFM. Se usa cuando no hay capitalización histórica.
- `ProveedorCapitalizacion` (Protocol): contrato inyectable
  `capitalizaciones(tickers, fecha) -> pd.Series` para suministrar capitalizaciones
  bursátiles reales point-in-time (p. ej. desde GrowSave o un proveedor de datos).
- `CapitalizacionEstatica`: proveedor trivial de un dict ticker -> capitalización
  (ignora la fecha); útil para el modelo Strategy `BLPriorReturnsModel(market_caps=...)`.
- `pesos_mercado`: usa el proveedor si existe y devuelve datos válidos; si no, el proxy
  por volumen; si tampoco hay volumen, equiponderado. Informa de la fuente usada.

Advertencia: el proxy por $-volumen NO es capitalización; sobrepondera activos muy
líquidos y baja capitalización. Es una aproximación aceptable solo si no hay
capitalización histórica (la BD del TFM no la tiene).

Se relaciona con: estimadores_mu.py (BL-prior y BL-momentum consumen w_mkt), rejilla.py
(la llama por celda y fecha) y portfolio_models.py (BLPriorReturnsModel / BLMomentumReturnsModel).
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MESES_VENTANA_VOLUMEN = 12


@runtime_checkable
class ProveedorCapitalizacion(Protocol):
    """Contrato de un proveedor de capitalización bursátil point-in-time."""

    def capitalizaciones(self, tickers: Sequence[str], fecha: pd.Timestamp) -> pd.Series:
        """Capitalización (misma moneda para todos) de cada ticker conocida en `fecha`; indexada por ticker."""
        ...


class CapitalizacionEstatica:
    """Proveedor de capitalizaciones fijas (no depende de la fecha): envoltorio de un dict ticker -> valor."""

    def __init__(self, capitalizaciones: Mapping[str, float]):
        self._capitalizaciones = dict(capitalizaciones)

    def capitalizaciones(self, tickers: Sequence[str], fecha: pd.Timestamp) -> pd.Series:
        """Devuelve la serie de capitalizaciones de `tickers` (NaN si falta alguno)."""
        return pd.Series({t: self._capitalizaciones.get(t, np.nan) for t in tickers}, dtype=float)


def _proxy_volumen(
    precios: pd.DataFrame,
    volumen: pd.DataFrame,
    activos: Sequence[str],
    fecha_fin: pd.Timestamp,
    meses: int,
) -> np.ndarray | None:
    """Pesos por $-volumen medio de los `meses` hasta `fecha_fin` (inclusive); None si no hay volumen."""
    fecha_ini = fecha_fin - pd.DateOffset(months=meses - 1)
    p = precios.loc[fecha_ini:fecha_fin, list(activos)]
    v = volumen.loc[fecha_ini:fecha_fin, list(activos)]
    dolar_volumen = (p * v).mean().clip(lower=0).fillna(0.0)
    total = float(dolar_volumen.sum())
    if total <= 0:
        return None
    return (dolar_volumen / total).values


def calcular_w_mkt_volumen(
    precios: pd.DataFrame,
    volumen: pd.DataFrame,
    activos: list,
    fecha_fin: pd.Timestamp,
    meses: int = MESES_VENTANA_VOLUMEN,
) -> np.ndarray:
    """
    Proxy de w_mkt por $-volumen medio móvil de los `meses` anteriores a `fecha_fin`
    (inclusive), usando solo precio y volumen ya observados (sin look-ahead).

    `fecha_fin` es normalmente t-1. Devuelve (N,) que suma 1; equiponderado si no hay
    $-volumen para ningún activo.
    """
    pesos = _proxy_volumen(precios, volumen, activos, fecha_fin, meses)
    if pesos is None:
        logger.debug("[w_mkt] Sin $-volumen disponible, usando equiponderado.")
        return np.full(len(activos), 1.0 / len(activos))
    return pesos


def pesos_mercado(
    activos: Sequence[str],
    fecha: pd.Timestamp,
    proveedor: ProveedorCapitalizacion | None = None,
    precios: pd.DataFrame | None = None,
    volumen: pd.DataFrame | None = None,
    meses: int = MESES_VENTANA_VOLUMEN,
) -> tuple[np.ndarray, str]:
    """
    Pesos de mercado (N,) de `activos` con información hasta `fecha` (normalmente t-1).

    Orden de preferencia:
    1. `proveedor.capitalizaciones(activos, fecha)`, si hay y todos los valores son
       finitos y positivos → fuente `"capitalizacion"`.
    2. Proxy por $-volumen (`precios` y `volumen`) → `"proxy_volumen"`.
    3. Equiponderado → `"equiponderado"`.

    Devuelve `(pesos, fuente)`; si el proveedor existe pero entrega datos inválidos se
    registra un aviso y se cae al siguiente nivel.
    """
    activos = list(activos)
    if proveedor is not None:
        caps = proveedor.capitalizaciones(activos, fecha).reindex(activos).astype(float)
        if np.all(np.isfinite(caps.values)) and np.all(caps.values > 0):
            return (caps / caps.sum()).values, "capitalizacion"
        logger.warning("[w_mkt] El proveedor de capitalización devolvió datos incompletos; se usa el proxy.")

    if precios is not None and volumen is not None:
        pesos = _proxy_volumen(precios, volumen, activos, pd.Timestamp(fecha), meses)
        if pesos is not None:
            return pesos, "proxy_volumen"

    return np.full(len(activos), 1.0 / len(activos)), "equiponderado"
