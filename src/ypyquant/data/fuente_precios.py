"""
Fuente de precios: la interfaz única por la que el motor, la API y el bot obtienen precios.

Responsabilidad:
- `FuentePrecios` (clase abstracta): INTERFAZ pequeña que oculta de dónde vienen los datos. Tres
  operaciones: `ohlcv` (serie ajustada o cruda), `identificador` (id estable del valor, distinto del
  símbolo de mercado, que puede cambiar o reutilizarse) y `deslistado` (si el valor dejó de cotizar,
  con fecha y, si se conoce, motivo). Cada fuente además es llamable `(ticker, inicio, fin)` y
  devuelve la serie ajustada: así sigue cumpliendo el contrato histórico `CargadorPrecios`
  (`data/matriz_precios.py`) y los llamadores antiguos no cambian.
- `Deslistado`: fecha del último día cotizado, motivo opcional y `inferido` (True si la fuente no lo
  sabe y lo deduce de los datos, como hace yfinance; False si es un dato explícito de la fuente).
- `FuenteDesdeCargador` / `como_fuente`: envuelve un callable suelto `(ticker, inicio, fin)` (tests,
  funciones de la aplicación) para que cumpla la interfaz; sus deslistados son inferidos.
- `inferir_deslistado`: la regla de inferencia compartida por los adaptadores que no saben de
  deslistados (yfinance, cargadores sueltos): el último dato queda más de `DIAS_PRECIO_CADUCADO`
  antes del final del rango que se pidió.

Contrato (lo verifica `tests/test_fuente_precios_contrato.py` contra todos los adaptadores):
- `ohlcv` -> DataFrame con índice DatetimeIndex ordenado, único y sin zona; columnas `Open, High, Low,
  Close, Volume, adjusted_close`. Con `ajuste="ajustado"`, `Close == adjusted_close` (dividendos y splits).
  Con `ajuste="crudo"`, `Close` es el precio tal como cotizó (hasta donde la fuente permita) y
  `adjusted_close` sigue siendo el ajustado. Devuelve None si no hay datos del símbolo.
- `identificador`: str no vacío, determinista y distinto para símbolos distintos.
- `deslistado`: None si el valor sigue cotizando (o la fuente no tiene datos); si no, la serie no pasa de
  `Deslistado.fecha`.

Adaptadores (carpeta `fuentes/`): `FuenteYFinanceCache` (producción hoy), `FuenteSintetica` (tests) y, en
el futuro, Norgate (ver `docs/planes/2026-11-22-fuente-precios-universos.md`).

Se relaciona con:
- ypyquant.data.matriz_precios (`CargadorPrecios`, contrato heredado que esta interfaz generaliza).
- ypyquant.data.fuentes (adaptadores y `fuente_por_defecto`) y ypyquant.data.universos (símbolos canónicos).
- ypyquant.estrategias.experimentos (`cargar_datos_experimento` recoge los deslistados) y .simulacion (los avisa).
- ypyquant.paper.simulador (llama a `refrescar_reciente` antes de avanzar las sesiones).
- pipelines/paper_trading/precios.py (el bot usa la misma fuente) y app/api (la construye al arrancar).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

Ajuste = Literal["ajustado", "crudo"]

# Días entre el último dato y el final del rango pedido a partir de los cuales se considera que el
# valor dejó de cotizar (fines de semana y festivos no cuentan como deslistado).
DIAS_PRECIO_CADUCADO = 5

COLUMNAS_OHLCV = ("Open", "High", "Low", "Close", "Volume", "adjusted_close")


@dataclass(frozen=True)
class Deslistado:
    """
    Un valor que dejó de cotizar.

    fecha: último día con cotización.
    motivo: causa si la fuente la conoce ('quiebra', 'fusion', 'exclusion'...); None si no.
    inferido: True si la fuente no sabe de deslistados y lo dedujo de que los datos terminan antes
        del final del rango pedido; False si es un dato explícito de la fuente.
    """

    fecha: pd.Timestamp
    motivo: str | None = None
    inferido: bool = False


def inferir_deslistado(ultima_fecha: pd.Timestamp, fin_cubierto: pd.Timestamp,
                       tolerancia_dias: int = DIAS_PRECIO_CADUCADO) -> Deslistado | None:
    """Deslistado inferido si los datos acaban más de `tolerancia_dias` antes de `fin_cubierto`; si no, None."""
    ultima, fin = pd.Timestamp(ultima_fecha), pd.Timestamp(fin_cubierto)
    if fin - ultima > pd.Timedelta(days=tolerancia_dias):
        return Deslistado(fecha=ultima, motivo=None, inferido=True)
    return None


class FuentePrecios(ABC):
    """
    Interfaz de una fuente de precios (ver la cabecera del módulo para el contrato).

    Los símbolos son los canónicos del proyecto (`ypyquant.data.universos`); cada adaptador traduce al
    formato de su proveedor si hace falta.
    """

    nombre: str = "fuente"

    @abstractmethod
    def ohlcv(self, simbolo: str, fecha_inicio: str, fecha_fin: str, ajuste: Ajuste = "ajustado") -> pd.DataFrame | None:
        """Serie OHLCV de `[fecha_inicio, fecha_fin]` ('YYYY-MM-DD', inclusivo) o None si no hay datos."""

    @abstractmethod
    def identificador(self, simbolo: str) -> str:
        """Identificador estable del valor (no cambia si el símbolo de mercado cambia o se reutiliza)."""

    @abstractmethod
    def deslistado(self, simbolo: str) -> Deslistado | None:
        """Información de deslistado del valor, o None si sigue cotizando (o no hay datos)."""

    def refrescar_reciente(self, simbolos: Sequence[str], hasta: str) -> list[str]:
        """
        Pone al día el tramo reciente de los símbolos (hasta `hasta`, 'YYYY-MM-DD') antes de que el paper interno
        pida los precios. Por defecto no hace nada (fuentes sin caché: sintética, cargadores sueltos). Devuelve los
        símbolos cuyo historial ajustado cambió (dividendo o split), que el paper interno re-basa. No altera los
        precios ya guardados salvo en ese caso: ver `FuenteYFinanceCache.refrescar_reciente`.
        """
        return []

    def __call__(self, ticker: str, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame | None:
        """Compatibilidad con `CargadorPrecios`: la serie ajustada."""
        return self.ohlcv(ticker, fecha_inicio, fecha_fin, "ajustado")


class FuenteDesdeCargador(FuentePrecios):
    """
    Adaptador de un callable suelto `(ticker, inicio, fin) -> OHLCV | None` a la interfaz.

    No sabe de cruda ni de identificadores reales (usa el símbolo) y sus deslistados son inferidos de
    lo que devolvió en las llamadas anteriores (último dato frente al final del rango pedido).
    """

    nombre = "cargador"

    def __init__(self, cargador: Callable[[str, str, str], pd.DataFrame | None]):
        self.cargador = cargador
        self._observado: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}

    def ohlcv(self, simbolo, fecha_inicio, fecha_fin, ajuste="ajustado"):
        if ajuste != "ajustado":
            raise ValueError("Este cargador suelto solo sirve precios ajustados")
        df = self.cargador(simbolo, fecha_inicio, fecha_fin)
        if df is not None and not df.empty:
            self._observado[simbolo] = (pd.Timestamp(df.index[-1]), pd.Timestamp(fecha_fin))
        return df

    def identificador(self, simbolo: str) -> str:
        return f"cargador:{simbolo}"

    def deslistado(self, simbolo: str) -> Deslistado | None:
        if simbolo not in self._observado:
            return None
        return inferir_deslistado(*self._observado[simbolo])


def como_fuente(origen: FuentePrecios | Callable[[str, str, str], pd.DataFrame | None]) -> FuentePrecios:
    """Devuelve `origen` si ya es una fuente; si es un callable suelto lo envuelve en `FuenteDesdeCargador`."""
    return origen if isinstance(origen, FuentePrecios) else FuenteDesdeCargador(origen)
