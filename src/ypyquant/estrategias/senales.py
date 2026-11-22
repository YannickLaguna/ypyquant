"""
Base para estrategias de entrada/salida (señales) dentro del registro de estrategias: largas, cortas o ambas.

Responsabilidad: dejar que el autor escriba REGLAS de entrada y salida por lado (tablas booleanas causales fecha x ticker)
y traducirlas a pesos objetivo. Es el adaptador entre el mundo pandas del registro y la máquina de estados pura
(`senales_maquina`): convierte las tablas del día en `Senal`, arma la vista de mercado (`Mercado`), llama a la máquina y
devuelve los pesos que entiende el motor (`BacktestEngine`, compartido con GrowSave, que no cambia). Backtest, barridos y
paper usan exactamente la misma `generar_pesos`.

Las tres piezas del marco (cada una cambiable sin tocar las otras):
- Reglas (esta clase): `senal_entrada`/`senal_salida` (lado largo) y `senal_entrada_corta`/`senal_salida_corta` (lado corto),
  más `indicadores`, `puntuacion`/`puntuacion_corta` (fuerza de la señal) y `salida_personalizada`.
- Gestión de la posición (`GestionPosicion`, en `senales_modelo`): stop de pérdida, take profit, trailing, tiempo máximo,
  enfriamiento y giro, simétricos para cortos.
- Dimensionamiento (`Dimensionamiento` y `LimitesExposicion`): peso fijo, inverso a la volatilidad, por fuerza de la señal
  o por riesgo por operación, con límites de exposición bruta y neta.

Clases:
- `ParametrosSenales`: parámetros comunes (máximo de posiciones, ponderación, stops, lados, límites de exposición...). Las
  subclases los extienden y la UI los convierte en formulario. Los parámetros posteriores a los experimentos guardados
  (`PARAMETROS_LARGO_CORTO_SENALES`) van fuera de la huella mientras valgan su defecto.
- `EstrategiaSenales`: el autor implementa las reglas de los lados que use; la base aporta la máquina de estados.

Modelo de ejecución (importante para interpretar los resultados):
- Las reglas y los stops se evalúan con el CIERRE de cada fecha de rebalanceo y la orden se ejecuta según el modo del motor
  (a ese mismo cierre en el backtest). No hay stops intradía ni gaps de apertura.
- Por defecto (`lados="largo"`) la estrategia es solo larga, como siempre. Con `lados="largo_corto"` o `"corto"` hay pesos
  NEGATIVOS (cortos) y las reglas de margen de `ypyquant.execution.margen` (exposición bruta máxima 2x, préstamo de
  títulos, llamadas de margen) se aplican en motor, paper y bot a partir de los mismos parámetros.
- El efectivo es el resto hasta 1 y se devuelve como el pseudo-ticker `CASH` (sin precio: el motor lo salta): mayor que 1 con
  cortos (el producto de las ventas) y NEGATIVO si hay préstamo de margen. `CASH` aparece en `pesos.parquet`.
- La frecuencia sugerida es `daily` porque los stops exigen mirar cada día; con una frecuencia menor las reglas solo se
  evalúan en las fechas de rebalanceo (el extremo del trailing sí cuenta los días intermedios).
- Estado JSON-serializable (`estado`): `posiciones` (con `lado`), `enfriamiento`, `fecha` y `pesos`; una instancia nueva con
  el estado cargado de JSON continúa igual (base de la reanudación del paper y del bot). El registro de operaciones con su
  motivo (`eventos`) vive en la instancia, no en el estado.

Rendimiento: `preparar(datos)` calcula una vez los indicadores y las tablas con TODO el histórico (válido porque las
operaciones son causales; lo comprueba `verificar_sin_lookahead`, que debe pasarse en el test de cada estrategia de
señales). Si `generar_pesos` se llama sin `preparar` (paper, donde `datos` crece cada día) la caché se recalcula cuando los
datos llegan más allá de lo precalculado.

Se relaciona con: ypyquant.estrategias.base (`Estrategia`, `ParametrosBase`, `matriz_cierres`), .senales_modelo (datos y
motivos), .senales_maquina (máquina de estados), .senales_tamano (tamaños), .registro (`registrar_estrategia`), .causalidad
(verificación de ausencia de look-ahead), .simulacion (etiqueta las operaciones con el motivo usando `eventos`),
.evaluacion (desglose por motivo), .senales_tecnicas (puente con las señales heredadas), execution/margen.py (reglas de
margen a partir de estos parámetros) y estrategias_usuario/ (plantillas).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from ypyquant.estrategias.base import Estrategia, ParametrosBase, matriz_cierres
from ypyquant.estrategias.senales_maquina import MaquinaPosiciones, Mercado
from ypyquant.estrategias.senales_modelo import (
    CORTO,
    LARGO,
    Dimensionamiento,
    GestionPosicion,
    LimitesExposicion,
    Senal,
)

TICKER_EFECTIVO = "CASH"  # peso sin precio: el motor lo salta y esa parte queda en caja
UMBRAL_EFECTIVO = 1e-9    # por debajo de este peso no se devuelve la clave CASH

Ponderacion = Literal["iguales", "inversa_volatilidad", "fuerza", "riesgo"]

# Parámetros posteriores a los experimentos guardados: `huella()` los omite mientras valgan su defecto (solo largo)
PARAMETROS_LARGO_CORTO_SENALES = (
    "lados", "max_posiciones_largas", "max_posiciones_cortas", "exposicion_bruta_max", "exposicion_neta_max",
    "exposicion_neta_min", "permitir_giro", "riesgo_por_operacion", "coste_prestamo_titulos", "diferencial_financiacion",
)


class ParametrosSenales(ParametrosBase):
    """Parámetros comunes de toda estrategia de señales (las subclases añaden los suyos)."""

    max_posiciones: int = Field(10, ge=1, le=200, description="Máximo de posiciones simultáneas (largas y cortas)")
    ponderacion: Ponderacion = Field(
        "iguales", description="Tamaño de cada entrada: 1/max_posiciones, inverso a su volatilidad, proporcional a la fuerza "
        "de la señal o por riesgo por operación (exige stop de pérdida)")
    peso_maximo: float = Field(0.25, gt=0, le=1, description="Tope de peso por activo al entrar")
    ventana_volatilidad: int = Field(20, ge=2, le=252, description="Sesiones para medir la volatilidad (ponderación inversa)")
    stop_perdida: float | None = Field(None, gt=0, lt=1, description="Sale si pierde esta fracción desde la entrada (en un corto, si sube)")
    take_profit: float | None = Field(None, gt=0, description="Sale si gana esta fracción desde la entrada (en un corto, si cae)")
    stop_trailing: float | None = Field(None, gt=0, lt=1, description="Sale si retrocede esta fracción desde su extremo favorable (máximo en largos, mínimo en cortos)")
    max_dias_posicion: int | None = Field(None, ge=1, description="Sale tras este número de sesiones en cartera")
    enfriamiento_dias: int = Field(5, ge=0, description="Días hábiles sin reentrar en un activo tras un stop de pérdida o trailing")
    mantener_deriva: bool = Field(True, description="Los pesos derivan con el precio entre rebalanceos (sin operar si no hay cambios)")
    # ---- largo-corto y dimensionamiento (posteriores a los experimentos guardados: fuera de la huella en su defecto) ----
    lados: Literal["largo", "corto", "largo_corto"] = Field(
        "largo", description="Lados que opera: solo largo (defecto), solo corto o ambos. Los cortos exigen cuenta de margen")
    max_posiciones_largas: int | None = Field(None, ge=1, le=200, description="Tope de posiciones largas (por defecto, solo el total)")
    max_posiciones_cortas: int | None = Field(None, ge=1, le=200, description="Tope de posiciones cortas (por defecto, solo el total)")
    exposicion_bruta_max: float = Field(1.0, gt=0, le=2.0, description="Exposición bruta máxima (suma de |pesos|); 1,6 permite un 130/30. Tope 2x (Reg T)")
    exposicion_neta_max: float = Field(1.0, gt=0, le=2.0, description="Exposición neta máxima (largos menos cortos); por encima de 1 hay préstamo de margen")
    exposicion_neta_min: float = Field(-1.0, ge=-2.0, le=1.0, description="Exposición neta mínima (0 = como mucho dólar-neutral; -1 = puede ir todo en corto)")
    permitir_giro: bool = Field(False, description="Con la señal contraria cierra y abre la posición opuesta el mismo día (si no, espera una sesión)")
    riesgo_por_operacion: float = Field(0.01, gt=0, le=0.2, description="Con ponderacion=riesgo: fracción del equity que se pierde si salta el stop de pérdida")
    coste_prestamo_titulos: float = Field(0.003, ge=0.0, le=0.5, description="Coste anual de pedir prestados los títulos de los cortos (0,003 = 0,3 %). Solo con cortos")
    diferencial_financiacion: float = Field(0.01, ge=0.0, le=0.1, description="Diferencial anual sobre la tasa libre del préstamo de margen. Solo con apalancamiento")

    @model_validator(mode="after")
    def _coherencia_largo_corto(self):
        if self.ponderacion == "riesgo" and self.stop_perdida is None:
            raise ValueError("ponderacion=riesgo exige stop_perdida (el peso es riesgo_por_operacion / stop_perdida)")
        if self.exposicion_neta_min > self.exposicion_neta_max:
            raise ValueError("exposicion_neta_min no puede superar a exposicion_neta_max")
        return self


@dataclass
class _Tablas:
    """
    Cálculos reutilizables de la estrategia (todo fecha x ticker): cierres, tablas de señales por lado y volatilidad.

    `reglas` lista `(lado, tipo, tabla booleana, tabla de fuerza o None)` de los lados activos; `volatilidad` solo existe
    con `ponderacion="inversa_volatilidad"`.
    """

    cierres: pd.DataFrame
    volatilidad: pd.DataFrame | None = None
    reglas: list[tuple[str, str, pd.DataFrame, pd.DataFrame | None]] = field(default_factory=list)


class EstrategiaSenales(Estrategia):
    """
    Estrategia de reglas de entrada y salida: el autor escribe las reglas, la base las convierte en pesos.

    Se implementan (todo vectorizado y CAUSAL sobre `cierres`, un DataFrame fecha x ticker) las reglas de los lados que se usen:
        - `senal_entrada(cierres, indicadores) -> DataFrame[bool]`: True = el ticker es candidato a comprar (largo).
        - `senal_salida(cierres, indicadores) -> DataFrame[bool]`: True = vender si está en cartera.
        - `senal_entrada_corta` y `senal_salida_corta`: lo mismo para vender en corto y cubrir (con `lados` corto o largo_corto).
        - opcionales: `indicadores(cierres, datos) -> dict[str, DataFrame]` (cálculos compartidos),
          `puntuacion` / `puntuacion_corta` (DataFrame[float]: fuerza de la señal; elige candidatos cuando hay más señales
          que huecos, mayor = antes, y con `ponderacion="fuerza"` reparte el tamaño; sin ella, orden alfabético) y
          `salida_personalizada(...)`.

    La base aplica, en este orden y en cada fecha: (1) deriva de los pesos con el precio, (2) salidas (fuera del universo,
    stop de pérdida, trailing, take profit, señal contraria, señal de salida, tiempo, regla propia), (3) entradas largas y
    cortas sobre los huecos libres y (4) completa con `CASH` hasta 1.

    Atributos de instancia extra:
        eventos: lista de `{fecha, ticker, tipo: 'entrada'|'salida', motivo, precio, lado}` en orden cronológico.
            Los motivos son los de `senales_modelo` (`entrada_larga`, `stop_loss`, `senal_contraria`...). `simular` la usa
            para etiquetar las operaciones con su motivo.
        maquina: la `MaquinaPosiciones` configurada con los parámetros (gestión, límites y dimensionamiento).
    """

    frecuencia_rebalanceo = "daily"
    parametros_fuera_de_huella_si_defecto = PARAMETROS_LARGO_CORTO_SENALES

    class Parametros(ParametrosSenales):
        """Parámetros de la base de señales (las subclases heredan de `ParametrosSenales`)."""

    def __init__(self, parametros: Any = None):
        super().__init__(parametros)
        self.eventos: list[dict[str, Any]] = []
        self._tablas: _Tablas | None = None
        self.maquina = self._construir_maquina()

    def _construir_maquina(self) -> MaquinaPosiciones:
        """Máquina de estados con la gestión, los límites y el dimensionamiento que dictan los parámetros."""
        p = self.parametros
        lados = {"largo": (LARGO,), "corto": (CORTO,), "largo_corto": (LARGO, CORTO)}[p.lados]
        gestion = GestionPosicion(
            stop_perdida=p.stop_perdida, take_profit=p.take_profit, stop_trailing=p.stop_trailing,
            max_dias_posicion=p.max_dias_posicion, enfriamiento_dias=p.enfriamiento_dias, permitir_giro=p.permitir_giro,
            mantener_deriva=p.mantener_deriva, lados=lados)
        limites = LimitesExposicion(
            max_posiciones=p.max_posiciones, max_largas=p.max_posiciones_largas, max_cortas=p.max_posiciones_cortas,
            exposicion_bruta_max=p.exposicion_bruta_max, exposicion_neta_max=p.exposicion_neta_max,
            exposicion_neta_min=p.exposicion_neta_min)
        dimension = Dimensionamiento(
            ponderacion=p.ponderacion, max_posiciones=p.max_posiciones, peso_maximo=p.peso_maximo,
            riesgo_por_operacion=p.riesgo_por_operacion, stop_perdida=p.stop_perdida)
        return MaquinaPosiciones(gestion, limites, dimension)

    def admite_cortos(self) -> bool:
        """True si la estrategia opera en corto (`lados` corto o largo_corto): los pesos pueden ser negativos."""
        return self.parametros.lados != "largo"

    # ---- lo que implementa el autor -------------------------------------------------------------------

    def indicadores(self, cierres: pd.DataFrame, datos: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Cálculos reutilizables (medias, RSI...) que reciben las reglas y `puntuacion`. Opcional."""
        return {}

    def senal_entrada(self, cierres: pd.DataFrame, indicadores: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Tabla booleana fecha x ticker: True donde la regla de COMPRA se cumple (causal: solo pasado y presente)."""
        return pd.DataFrame(False, index=cierres.index, columns=cierres.columns)

    def senal_salida(self, cierres: pd.DataFrame, indicadores: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Tabla booleana fecha x ticker: True donde la regla de VENTA de un largo se cumple (causal)."""
        return pd.DataFrame(False, index=cierres.index, columns=cierres.columns)

    def senal_entrada_corta(self, cierres: pd.DataFrame, indicadores: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Tabla booleana: True donde la regla de VENTA EN CORTO se cumple (solo con `lados` corto o largo_corto)."""
        return pd.DataFrame(False, index=cierres.index, columns=cierres.columns)

    def senal_salida_corta(self, cierres: pd.DataFrame, indicadores: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Tabla booleana: True donde la regla de RECOMPRA (cubrir un corto) se cumple."""
        return pd.DataFrame(False, index=cierres.index, columns=cierres.columns)

    def puntuacion(self, cierres: pd.DataFrame, indicadores: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        """Fuerza de las entradas largas (mayor = antes y, con `ponderacion="fuerza"`, más tamaño). Opcional."""
        return None

    def puntuacion_corta(self, cierres: pd.DataFrame, indicadores: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        """Fuerza de las entradas cortas (mayor = antes y más tamaño). Opcional."""
        return None

    def salida_personalizada(self, ticker: str, fecha: datetime, posicion: dict[str, Any], cierres_hasta: pd.Series) -> bool:
        """Regla de salida que depende de la posición (`lado`, `fecha_entrada`, `precio_entrada`, `maximo`/`minimo`, `peso`). Opcional."""
        return False

    # ---- caché de tablas ----------------------------------------------------------------------------

    def preparar(self, datos: dict[str, pd.DataFrame]) -> None:
        """Calcula una vez indicadores y tablas con todo el histórico (permitido porque las reglas son causales)."""
        self._calcular(datos)

    def _calcular(self, datos: dict[str, pd.DataFrame]) -> None:
        cierres = matriz_cierres(datos)
        indicadores = self.indicadores(cierres, datos)
        lados = self.maquina.gestion.lados
        tablas = _Tablas(cierres)
        if LARGO in lados:
            entrada = self._validar_tabla("senal_entrada", self.senal_entrada(cierres, indicadores), cierres)
            salida = self._validar_tabla("senal_salida", self.senal_salida(cierres, indicadores), cierres)
            puntuacion = self._validar_puntuacion("puntuacion", self.puntuacion(cierres, indicadores), cierres)
            tablas.reglas += [(LARGO, "entrada", entrada, puntuacion), (LARGO, "salida", salida, None)]
        if CORTO in lados:
            entrada = self._validar_tabla("senal_entrada_corta", self.senal_entrada_corta(cierres, indicadores), cierres)
            salida = self._validar_tabla("senal_salida_corta", self.senal_salida_corta(cierres, indicadores), cierres)
            puntuacion = self._validar_puntuacion("puntuacion_corta", self.puntuacion_corta(cierres, indicadores), cierres)
            tablas.reglas += [(CORTO, "entrada", entrada, puntuacion), (CORTO, "salida", salida, None)]
        if self.parametros.ponderacion == "inversa_volatilidad":
            tablas.volatilidad = cierres.pct_change(fill_method=None).rolling(self.parametros.ventana_volatilidad).std()
        self._tablas = tablas

    def _validar_puntuacion(self, nombre: str, tabla: Any, cierres: pd.DataFrame) -> pd.DataFrame | None:
        """Puntuación opcional: None se respeta; si existe debe tener la forma de los cierres."""
        return None if tabla is None else self._validar_tabla(nombre, tabla, cierres, booleana=False)

    @staticmethod
    def _validar_tabla(nombre: str, tabla: Any, cierres: pd.DataFrame, booleana: bool = True) -> pd.DataFrame:
        """La tabla debe ser un DataFrame con las mismas fechas y tickers que `cierres`; las señales se fuerzan a bool."""
        if not isinstance(tabla, pd.DataFrame) or not tabla.index.equals(cierres.index) \
                or set(tabla.columns) != set(cierres.columns):
            raise ValueError(f"{nombre} debe devolver un DataFrame con las mismas fechas y tickers que `cierres` "
                             f"(forma esperada {cierres.shape}, recibida {getattr(tabla, 'shape', type(tabla).__name__)})")
        tabla = tabla[cierres.columns]
        return tabla.fillna(False).astype(bool) if booleana else tabla.astype(float)

    def _tablas_para(self, datos: dict[str, pd.DataFrame]) -> _Tablas:
        """Tablas vigentes; se recalculan si los datos llegan más allá de la caché o traen tickers nuevos."""
        tablas = self._tablas
        if tablas is not None and datos:
            ultima = max(df.index[-1] for df in datos.values() if len(df))
            if tablas.cierres.index[-1] >= ultima and set(datos) <= set(tablas.cierres.columns):
                return tablas
        self._calcular(datos)
        return self._tablas  # type: ignore[return-value]

    # ---- adaptador a la máquina de estados ------------------------------------------------------------

    def generar_pesos(self, fecha: datetime, datos: dict[str, pd.DataFrame], estado: dict[str, Any]) -> dict[str, float]:
        """
        Pesos objetivo en `fecha`: la máquina aplica deriva, salidas y entradas sobre `estado` y se completa con `CASH`.

        Idempotente: una segunda llamada con la misma fecha devuelve los mismos pesos sin tocar el estado.
        """
        clave = f"{pd.Timestamp(fecha):%Y-%m-%d}"
        if estado.get("fecha") == clave and "pesos" in estado:
            return dict(estado["pesos"])
        tablas = self._tablas_para(datos)
        fila = int(tablas.cierres.index.searchsorted(pd.Timestamp(fecha), side="right")) - 1
        if fila < 0:
            return {TICKER_EFECTIVO: 1.0}
        fila_previa = None
        if estado.get("fecha"):
            fila_previa = int(tablas.cierres.index.searchsorted(pd.Timestamp(estado["fecha"]), side="right")) - 1
        posiciones = copy.deepcopy(estado.get("posiciones", {}))
        enfriamiento = {t: f for t, f in estado.get("enfriamiento", {}).items() if f > clave}

        mercado = self._mercado(tablas, fila, fila_previa, fecha, clave, frozenset(datos))
        eventos = self.maquina.avanzar(posiciones, enfriamiento, self._senales_del_dia(tablas, fila), mercado)

        pesos = MaquinaPosiciones.pesos(posiciones)
        efectivo = 1.0 - sum(pesos.values())
        if not pesos:
            pesos[TICKER_EFECTIVO] = 1.0
        elif efectivo > UMBRAL_EFECTIVO or (efectivo < -UMBRAL_EFECTIVO and self.maquina.limites.admite_prestamo):
            pesos[TICKER_EFECTIVO] = efectivo  # resto hasta 1: > 1 con cortos (producto de la venta), < 0 con préstamo
        estado.update(fecha=clave, pesos=pesos, posiciones=posiciones, enfriamiento=enfriamiento)
        self.eventos.extend(eventos)
        return dict(pesos)

    @staticmethod
    def _precio(tablas: _Tablas, ticker: str, fila: int) -> float | None:
        """Cierre del ticker en la fila (None si no hay columna o el valor no es finito)."""
        if ticker not in tablas.cierres.columns:
            return None
        valor = tablas.cierres[ticker].iloc[fila]
        return float(valor) if math.isfinite(valor) and valor > 0 else None

    @staticmethod
    def _senales_del_dia(tablas: _Tablas, fila: int) -> list[Senal]:
        """Las tablas de reglas de la fila como `Senal` (solo donde la regla es True); la fuerza viene de la puntuación."""
        columnas = tablas.cierres.columns
        senales: list[Senal] = []
        for lado, tipo, tabla, puntuacion in tablas.reglas:
            posiciones = np.flatnonzero(tabla.iloc[fila].to_numpy())
            fuerzas = puntuacion.iloc[fila].to_numpy() if puntuacion is not None and len(posiciones) else None
            for i in posiciones:
                senales.append(Senal(columnas[i], lado, tipo, float(fuerzas[i]) if fuerzas is not None else None))
        return senales

    def _mercado(self, tablas: _Tablas, fila: int, fila_previa: int | None, fecha: datetime, dia: str,
                 universo: frozenset[str]) -> Mercado:
        """Vista del mercado en la fecha (datos hasta ella) que consume la máquina de estados."""
        cierres = tablas.cierres
        hay_previa = fila_previa is not None and fila_previa >= 0

        def crecimiento(ticker: str) -> float:
            antes, ahora = self._precio(tablas, ticker, fila_previa), self._precio(tablas, ticker, fila)
            return ahora / antes if antes and ahora else 1.0

        def extremos_tramo(ticker: str) -> tuple[float, float] | None:
            if ticker not in cierres.columns:
                return None
            tramo = cierres[ticker].iloc[fila_previa + 1: fila + 1].dropna()
            return (float(tramo.max()), float(tramo.min())) if len(tramo) else None

        def sesiones_desde(fecha_entrada: str) -> int:
            return fila - int(cierres.index.searchsorted(pd.Timestamp(fecha_entrada), side="left"))

        def volatilidad(ticker: str) -> float | None:
            return None if tablas.volatilidad is None else float(tablas.volatilidad[ticker].iloc[fila])

        def regla_propia(ticker: str, posicion: dict[str, Any]) -> bool:
            return self.salida_personalizada(ticker, fecha, posicion, cierres[ticker].iloc[: fila + 1])

        return Mercado(
            fecha=dia, universo=universo, precio=lambda ticker: self._precio(tablas, ticker, fila), hay_previa=hay_previa,
            crecimiento=crecimiento, extremos_tramo=extremos_tramo, sesiones_desde=sesiones_desde,
            volatilidad=volatilidad, regla_propia=regla_propia)
