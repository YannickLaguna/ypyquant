"""
Estrategia base: contrato para escribir estrategias de portafolio en Python.

Responsabilidad:
- `Estrategia`: clase abstracta. Una estrategia declara `nombre`, `descripcion`, una subclase
  Pydantic `Parametros` (defaults, límites y descripciones; de ahí sale el formulario de la UI)
  e implementa `generar_pesos(fecha, datos, estado)`. Ganchos opcionales: `preparar(datos)` y
  `frecuencia_rebalanceo` (sugerida; el experimento puede sobrescribirla).
- `Estrategia.sesiones_necesarias()`: sesiones diarias de historia que necesita según sus parámetros
  (el paper trading deriva de ahí cuántos días descargar; ver `MARGEN_SESIONES`).
- `Estrategia.como_funcion()`: adaptador a la firma `FuncionEstrategia` que espera
  `BacktestEngine.run_backtest` (`(fecha, datos) -> pesos`). Recorta los datos a `<= fecha`
  (sin look-ahead), aplica opcionalmente un filtro de universo point-in-time (`filtro_universo`),
  mantiene el `estado` entre rebalanceos, sanea los pesos y registra el
  historial de pesos objetivo (lo usa el experimento para guardar `pesos.parquet` y el turnover).
- Utilidades puras para escribir estrategias: `matriz_cierres`, `matriz_volumen`, `recortar_hasta`.

Se relaciona con:
- ypyquant.estrategias.simulacion (inyecta `semilla` y `rng` en cada ejecución).
- ypyquant.estrategias.registro (registro y descubrimiento), .incluidas (estrategias de serie),
  .experimentos (ejecuta una estrategia con el motor).
- ypyquant.portfolio.backtest_engine (FuncionEstrategia, BacktestEngine).
- ypyquant.data.matriz_precios (extraer_serie_precios).
- ypyquant.estrategias.universo_dinamico (fabrica el `FiltroUniverso`) y .causalidad (lo prueba).
- estrategias_usuario/ (en la raíz del repo): estrategias propias que heredan de esta clase.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from ypyquant.data.matriz_precios import extraer_serie_precios
from ypyquant.execution.calendario import FRECUENCIAS_VALIDAS  # noqa: F401  (reexportada: única definición)
from ypyquant.portfolio.backtest_engine import FuncionEstrategia

# `(fecha, datos_recortados_a_fecha) -> {ticker: OHLCV}` solo con los tickers elegibles en esa fecha
# (universo point-in-time; ver `ypyquant.estrategias.universo_dinamico`).
FiltroUniverso = Callable[[datetime, dict[str, pd.DataFrame]], dict[str, pd.DataFrame]]

# Sesiones extra que se piden por encima del mínimo estricto de una estrategia (festivos, huecos de
# datos y el último día incompleto): `Estrategia.sesiones_necesarias` las suma a su ventana.
MARGEN_SESIONES = 10


class ParametrosBase(BaseModel):
    """Base de los parámetros de una estrategia: se rechazan campos desconocidos (evita erratas en la UI/API)."""

    model_config = ConfigDict(extra="forbid")


def recortar_hasta(datos: dict[str, pd.DataFrame], fecha: datetime) -> dict[str, pd.DataFrame]:
    """Devuelve `{ticker: OHLCV hasta fecha inclusive}` sin copiar (vistas); los tickers sin datos se omiten."""
    recortados: dict[str, pd.DataFrame] = {}
    for ticker, df in datos.items():
        fin = df.index.searchsorted(pd.Timestamp(fecha), side="right")
        if fin > 0:
            recortados[ticker] = df.iloc[:fin]
    return recortados


def matriz_cierres(datos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Matriz fecha x ticker de cierres ajustados (tickers sin datos omitidos)."""
    series = {}
    for ticker, df in datos.items():
        serie = extraer_serie_precios(df, ticker, avisar=False)
        if serie is not None and not serie.empty:
            series[ticker] = serie
    return pd.DataFrame(series).sort_index()


def matriz_volumen(datos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Matriz fecha x ticker de volumen (acepta columnas 'Volume' o 'volume'); vacía si no hay."""
    series = {}
    for ticker, df in datos.items():
        for columna in ("Volume", "volume"):
            if columna in df.columns:
                series[ticker] = df[columna]
                break
    return pd.DataFrame(series).sort_index()


def sanear_pesos(pesos: dict[str, float]) -> dict[str, float]:
    """
    Deja solo pesos finitos y positivos (el motor es long-only) y normaliza a suma 1.

    Si la suma es 0 devuelve `{}` (el motor liquida y queda en efectivo). El motor ya
    normaliza a 1 cuando la suma se desvía más de 1 %, así que no existe efectivo parcial.
    """
    limpios = {t: float(w) for t, w in pesos.items() if w is not None and math.isfinite(w) and w > 0}
    total = sum(limpios.values())
    if total <= 0:
        return {}
    return {t: w / total for t, w in limpios.items()}


class Estrategia(ABC):
    """
    Estrategia de portafolio: decide los pesos objetivo en cada fecha de rebalanceo.

    Cómo escribir una (ver también docs/ui.md y estrategias_usuario/):

        @registrar_estrategia
        class MiEstrategia(Estrategia):
            nombre = "mi_estrategia"
            descripcion = "Qué hace, en una frase."

            class Parametros(ParametrosBase):
                ventana: int = Field(60, ge=5, le=500, description="Días de la ventana")

            def generar_pesos(self, fecha, datos, estado):
                cierres = matriz_cierres(datos)
                ...
                return {"AAPL": 0.5, "MSFT": 0.5}

    Atributos de clase:
        nombre: identificador único (minúsculas y `_`).
        descripcion: texto corto que se muestra en la UI.
        frecuencia_rebalanceo: frecuencia sugerida ('daily'|'weekly'|'monthly'|'quarterly').
        universo_sugerido: nombre de universo que la UI propone por defecto (`'sp500_pit'` en las de sección
            cruzada, para evitar el sesgo de supervivencia); None = sin sugerencia.
        Parametros: modelo Pydantic con los parámetros (sin campos si no hay).
        parametros_fuera_de_huella_si_defecto: nombres de parámetros añadidos DESPUÉS de que existieran experimentos
            guardados: `ConfiguracionExperimento.huella()` los omite mientras valgan su defecto, para que la huella
            (y con ella el id) de los experimentos ya guardados no cambie.
        origen / archivo: los rellena el registro ('incluida' | 'usuario' y ruta del .py).

    Atributos de instancia:
        parametros: instancia validada de `Parametros`.
        estado: dict mutable que persiste entre rebalanceos de una misma simulación; el
            adaptador guarda en `estado['pesos_previos']` los últimos pesos objetivo.
        historial_pesos: lista `(fecha, pesos)` de todos los rebalanceos pedidos al motor.
        semilla: semilla del experimento (la fija `simular`).
        rng: generador de números aleatorios PROPIO de esta ejecución (`np.random.Generator`); `simular` se lo
            pasa explícitamente. Úsalo (vía `generador_azar()`) para todo azar: ya no se siembra el estado global
            de numpy/random, así dos experimentos en hilos distintos no interfieren.
        avisos: mensajes no fatales (sin duplicados) que el experimento añade a los suyos.
        errores: mensajes de las excepciones de `generar_pesos` (el motor las traga y sigue;
            el experimento falla si no hubo ningún rebalanceo válido).
    """

    nombre: ClassVar[str]
    descripcion: ClassVar[str] = ""
    frecuencia_rebalanceo: ClassVar[str] = "monthly"
    universo_sugerido: ClassVar[str | None] = None
    parametros_fuera_de_huella_si_defecto: ClassVar[tuple[str, ...]] = ()
    origen: ClassVar[str] = "incluida"
    archivo: ClassVar[str | None] = None

    class Parametros(ParametrosBase):
        """Parámetros de la estrategia (sin parámetros por defecto)."""

    def __init__(self, parametros: BaseModel | dict[str, Any] | None = None):
        if parametros is None:
            parametros = {}
        if isinstance(parametros, dict):
            parametros = self.Parametros(**parametros)
        self.parametros = parametros
        self.estado: dict[str, Any] = {}
        self.historial_pesos: list[tuple[datetime, dict[str, float]]] = []
        self.semilla = 0
        self.rng: np.random.Generator | None = None
        self.errores: list[str] = []
        self.avisos: list[str] = []  # avisos no fatales (p. ej. pocas observaciones); el experimento los recoge

    def generador_azar(self) -> np.random.Generator:
        """
        Generador aleatorio de esta ejecución: el que inyectó `simular` o, si no hay, `default_rng(semilla)`.

        Se crea una sola vez y se comparte entre rebalanceos (el flujo avanza como un único estado).
        """
        if self.rng is None:
            self.rng = np.random.default_rng(self.semilla)
        return self.rng

    def preparar(self, datos: dict[str, pd.DataFrame]) -> None:
        """
        Gancho opcional, llamado una vez antes del primer rebalanceo con TODO el histórico cargado.

        Útil para cálculos causales costosos (p. ej. indicadores rolling). Ojo: `datos` incluye
        fechas futuras; no uses nada que mire hacia delante.
        """

    @abstractmethod
    def generar_pesos(self, fecha: datetime, datos: dict[str, pd.DataFrame], estado: dict[str, Any]) -> dict[str, float]:
        """
        Pesos objetivo `{ticker: peso}` en `fecha`.

        Args:
            fecha: fecha de rebalanceo.
            datos: `{ticker: OHLCV}` recortado a `<= fecha` (columnas 'Close', 'adjusted_close', 'Volume'...).
            estado: dict persistente entre llamadas (memoria de la estrategia).
        """

    def sesiones_necesarias(self) -> int:
        """
        Sesiones diarias de historia que necesita la estrategia para no caer en su caso de reserva
        (p. ej. pesos iguales por falta de datos), según sus parámetros y con `MARGEN_SESIONES` de holgura.

        0 (por defecto) = la estrategia no declara requisito. Lo usa el bot de paper trading para decidir
        cuántos días descargar; las estrategias con ventana (momentum, estimadores) lo sobrescriben.
        """
        return 0

    def como_funcion(self, filtro_universo: FiltroUniverso | None = None) -> FuncionEstrategia:
        """
        Adaptador a la firma del motor: `(fecha, datos_completos) -> pesos` con recorte, estado y saneado.

        `filtro_universo` (opcional, universo point-in-time): `(fecha, datos_recortados) -> datos` que deja
        a `generar_pesos` solo los tickers elegibles en cada fecha. `preparar` sigue viendo el histórico
        completo de todos los tickers cargados. Sin filtro el comportamiento es el de siempre.
        """
        preparada = False

        def funcion(fecha: datetime, datos: dict[str, pd.DataFrame]) -> dict[str, float]:
            nonlocal preparada
            if not preparada:
                self.preparar(datos)
                preparada = True
            try:
                recortados = recortar_hasta(datos, fecha)
                if filtro_universo is not None:
                    recortados = filtro_universo(fecha, recortados)
                pesos = sanear_pesos(self.generar_pesos(fecha, recortados, self.estado))
            except Exception as error:  # el motor la traga; la dejamos anotada para el experimento
                self.errores.append(f"{fecha:%Y-%m-%d}: {type(error).__name__}: {error}")
                raise
            self.estado["pesos_previos"] = pesos
            self.historial_pesos.append((fecha, pesos))
            return pesos

        return funcion
