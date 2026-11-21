"""
Base para estrategias de entrada/salida (señales) dentro del registro de estrategias.

Responsabilidad: dejar que el usuario escriba REGLAS de compra y venta (tablas booleanas causales
fecha x ticker) y traducirlas a pesos objetivo con una máquina de estados persistente (posiciones,
stops, huecos libres, enfriamiento y efectivo). Así backtest, barridos y paper usan exactamente la
misma `generar_pesos` y el motor (`BacktestEngine`, compartido con GrowSave) no cambia: sigue
recibiendo pesos.

Clases:
- `ParametrosSenales`: parámetros comunes de la base (máximo de posiciones, ponderación, tope por
  activo, stops, tiempo máximo, enfriamiento...). Las subclases los extienden y la UI los convierte
  en formulario.
- `EstrategiaSenales`: el usuario implementa `senal_entrada` y `senal_salida` (y, opcionalmente,
  `indicadores`, `puntuacion`, `salida_personalizada`); la base aporta la máquina de estados.

Modelo de ejecución (importante para interpretar los resultados):
- Las reglas y los stops se evalúan con el CIERRE de cada fecha de rebalanceo y la orden se ejecuta
  a ese mismo cierre (como el resto de estrategias del registro). No hay stops intradía ni gaps de apertura.
- El motor es long-only: solo hay compras y ventas, no cortos.
- El efectivo no invertido se devuelve como el pseudo-ticker `CASH` (sin precio: el motor lo salta y esa
  parte queda en caja; ver tests/test_backtest_motor.py). `CASH` aparece en `pesos.parquet`.
- La frecuencia sugerida es `daily` porque los stops exigen mirar cada día; con una frecuencia menor
  las reglas solo se evalúan en las fechas de rebalanceo (el máximo del trailing sí cuenta los días intermedios).
- Estado JSON-serializable (`estado`): `posiciones`, `enfriamiento`, `fecha` y `pesos`; una instancia
  nueva con el estado cargado de JSON continúa igual (base de la reanudación del paper). El registro
  de operaciones con su motivo (`eventos`) vive en la instancia, no en el estado.

Rendimiento: `preparar(datos)` calcula una vez los indicadores y las tablas con TODO el histórico
(válido porque las operaciones son causales; lo comprueba `verificar_sin_lookahead`, que debe pasarse en
el test de cada estrategia de señales). Si `generar_pesos` se llama sin `preparar` (paper, donde `datos`
crece cada día) la caché se recalcula cuando los datos llegan más allá de lo precalculado.

Se relaciona con: ypyquant.estrategias.base (`Estrategia`, `ParametrosBase`, `matriz_cierres`),
.registro (`registrar_estrategia`), .causalidad (verificación de ausencia de look-ahead), .experimentos
(añade la columna `motivo` a las operaciones usando `eventos`), .senales_tecnicas (puente con las señales
heredadas) y estrategias_usuario/ (ejemplos `cruce_medias` y `canal_rsi_volatilidad`).
"""

from __future__ import annotations

import copy
import math
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from ypyquant.estrategias.base import Estrategia, ParametrosBase, matriz_cierres

TICKER_EFECTIVO = "CASH"  # peso sin precio: el motor lo salta y esa parte queda en caja
UMBRAL_EFECTIVO = 1e-9    # por debajo de este peso no se devuelve la clave CASH
MOTIVOS_CON_ENFRIAMIENTO = ("stop_perdida", "stop_trailing")  # tras una salida de pérdida se espera antes de reentrar


class ParametrosSenales(ParametrosBase):
    """Parámetros comunes de toda estrategia de señales (las subclases añaden los suyos)."""

    max_posiciones: int = Field(10, ge=1, le=200, description="Máximo de posiciones simultáneas")
    ponderacion: Literal["iguales", "inversa_volatilidad"] = Field(
        "iguales", description="Tamaño de cada entrada: 1/max_posiciones, o inversamente proporcional a su volatilidad")
    peso_maximo: float = Field(0.25, gt=0, le=1, description="Tope de peso por activo al entrar")
    ventana_volatilidad: int = Field(20, ge=2, le=252, description="Sesiones para medir la volatilidad (ponderación inversa)")
    stop_perdida: float | None = Field(None, gt=0, lt=1, description="Sale si pierde esta fracción desde la entrada")
    take_profit: float | None = Field(None, gt=0, description="Sale si gana esta fracción desde la entrada")
    stop_trailing: float | None = Field(None, gt=0, lt=1, description="Sale si cae esta fracción desde el máximo desde la entrada")
    max_dias_posicion: int | None = Field(None, ge=1, description="Sale tras este número de sesiones en cartera")
    enfriamiento_dias: int = Field(5, ge=0, description="Días hábiles sin reentrar en un activo tras un stop de pérdida o trailing")
    mantener_deriva: bool = Field(True, description="Los pesos derivan con el precio entre rebalanceos (sin operar si no hay cambios)")


@dataclass
class _Tablas:
    """Cálculos reutilizables de la estrategia: cierres, indicadores y tablas de señales (todo fecha x ticker)."""

    cierres: pd.DataFrame
    entrada: pd.DataFrame
    salida: pd.DataFrame
    puntuacion: pd.DataFrame | None
    volatilidad: pd.DataFrame | None


class EstrategiaSenales(Estrategia):
    """
    Estrategia de reglas de entrada y salida: el usuario escribe las reglas, la base las convierte en pesos.

    Se implementan (todo vectorizado y CAUSAL sobre `cierres`, un DataFrame fecha x ticker):
        - `senal_entrada(cierres, indicadores) -> DataFrame[bool]`: True = el ticker es candidato a comprar.
        - `senal_salida(cierres, indicadores) -> DataFrame[bool]`: True = vender si está en cartera.
        - opcionales: `indicadores(cierres, datos) -> dict[str, DataFrame]` (cálculos compartidos),
          `puntuacion(cierres, indicadores) -> DataFrame[float]` (elige candidatos cuando hay más
          señales que huecos; sin ella, orden alfabético) y `salida_personalizada(...)`.

    La base aplica, en este orden y en cada fecha: (1) deriva de los pesos con el precio, (2) salidas
    (fuera del universo, stop de pérdida, trailing, take profit, señal, tiempo, regla propia),
    (3) entradas sobre los huecos libres y (4) completa con `CASH` hasta 1.

    Atributos de instancia extra:
        eventos: lista de `{fecha, ticker, tipo: 'entrada'|'salida', motivo, precio}` en orden cronológico.
            `ejecutar_con_datos` la usa para etiquetar las operaciones con su motivo.
    """

    frecuencia_rebalanceo = "daily"

    class Parametros(ParametrosSenales):
        """Parámetros de la base de señales (las subclases heredan de `ParametrosSenales`)."""

    def __init__(self, parametros: Any = None):
        super().__init__(parametros)
        self.eventos: list[dict[str, Any]] = []
        self._tablas: _Tablas | None = None

    # ---- lo que implementa el usuario ---------------------------------------------------------------

    def indicadores(self, cierres: pd.DataFrame, datos: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Cálculos reutilizables (medias, RSI...) que reciben `senal_entrada`, `senal_salida` y `puntuacion`. Opcional."""
        return {}

    @abstractmethod
    def senal_entrada(self, cierres: pd.DataFrame, indicadores: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Tabla booleana fecha x ticker: True donde la regla de compra se cumple (causal: solo pasado y presente)."""

    @abstractmethod
    def senal_salida(self, cierres: pd.DataFrame, indicadores: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Tabla booleana fecha x ticker: True donde la regla de venta se cumple (causal)."""

    def puntuacion(self, cierres: pd.DataFrame, indicadores: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        """Tabla numérica para ordenar candidatos cuando hay más entradas que huecos (mayor = antes). Opcional."""
        return None

    def salida_personalizada(self, ticker: str, fecha: datetime, posicion: dict[str, Any], cierres_hasta: pd.Series) -> bool:
        """Regla de salida que depende de la posición (`fecha_entrada`, `precio_entrada`, `maximo`, `peso`). Opcional."""
        return False

    # ---- caché de tablas ----------------------------------------------------------------------------

    def preparar(self, datos: dict[str, pd.DataFrame]) -> None:
        """Calcula una vez indicadores y tablas con todo el histórico (permitido porque las reglas son causales)."""
        self._calcular(datos)

    def _calcular(self, datos: dict[str, pd.DataFrame]) -> None:
        cierres = matriz_cierres(datos)
        indicadores = self.indicadores(cierres, datos)
        entrada = self._validar_tabla("senal_entrada", self.senal_entrada(cierres, indicadores), cierres)
        salida = self._validar_tabla("senal_salida", self.senal_salida(cierres, indicadores), cierres)
        puntuacion = self.puntuacion(cierres, indicadores)
        if puntuacion is not None:
            puntuacion = self._validar_tabla("puntuacion", puntuacion, cierres, booleana=False)
        volatilidad = None
        if self.parametros.ponderacion == "inversa_volatilidad":
            volatilidad = cierres.pct_change(fill_method=None).rolling(self.parametros.ventana_volatilidad).std()
        self._tablas = _Tablas(cierres, entrada, salida, puntuacion, volatilidad)

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

    # ---- máquina de estados -------------------------------------------------------------------------

    def generar_pesos(self, fecha: datetime, datos: dict[str, pd.DataFrame], estado: dict[str, Any]) -> dict[str, float]:
        """
        Pesos objetivo en `fecha`: aplica deriva, salidas y entradas sobre `estado` y completa con `CASH`.

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
        eventos: list[dict[str, Any]] = []

        self._aplicar_deriva(posiciones, tablas, fila, fila_previa)
        vendidos = self._aplicar_salidas(posiciones, enfriamiento, tablas, fila, fecha, set(datos), eventos)
        self._aplicar_entradas(posiciones, enfriamiento, vendidos, tablas, fila, fecha, set(datos), eventos)

        pesos = {t: p["peso"] for t, p in sorted(posiciones.items())}
        efectivo = 1.0 - sum(pesos.values())
        if efectivo > UMBRAL_EFECTIVO or not pesos:
            pesos[TICKER_EFECTIVO] = max(efectivo, 0.0) if pesos else 1.0
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

    def _aplicar_deriva(self, posiciones: dict[str, Any], tablas: _Tablas, fila: int, fila_previa: int | None) -> None:
        """Actualiza el máximo de cada posición y, con `mantener_deriva`, sus pesos según el precio desde la llamada anterior."""
        if fila_previa is None or fila_previa < 0 or not posiciones:
            return
        crecimiento = {}
        for ticker, posicion in posiciones.items():
            if ticker in tablas.cierres.columns:
                tramo = tablas.cierres[ticker].iloc[fila_previa + 1: fila + 1].dropna()
                if len(tramo):
                    posicion["maximo"] = max(posicion["maximo"], float(tramo.max()))
            antes, ahora = self._precio(tablas, ticker, fila_previa), self._precio(tablas, ticker, fila)
            crecimiento[ticker] = ahora / antes if antes and ahora else 1.0
        if not self.parametros.mantener_deriva:
            return
        efectivo = max(1.0 - sum(p["peso"] for p in posiciones.values()), 0.0)
        total = efectivo + sum(p["peso"] * crecimiento[t] for t, p in posiciones.items())
        for ticker, posicion in posiciones.items():
            posicion["peso"] = posicion["peso"] * crecimiento[ticker] / total

    def _motivo_de_salida(self, ticker: str, posicion: dict[str, Any], tablas: _Tablas, fila: int, fecha: datetime,
                          universo: set[str]) -> str | None:
        """Primer motivo por el que la posición debe cerrarse (orden: universo, stops, take profit, señal, tiempo, regla propia)."""
        p = self.parametros
        precio = self._precio(tablas, ticker, fila)
        if ticker not in universo:
            return "fuera_universo"
        if precio is not None:
            if p.stop_perdida is not None and precio / posicion["precio_entrada"] - 1.0 <= -p.stop_perdida:
                return "stop_perdida"
            if p.stop_trailing is not None and precio / posicion["maximo"] - 1.0 <= -p.stop_trailing:
                return "stop_trailing"
            if p.take_profit is not None and precio / posicion["precio_entrada"] - 1.0 >= p.take_profit:
                return "take_profit"
        if bool(tablas.salida[ticker].iloc[fila]):
            return "senal"
        if p.max_dias_posicion is not None:
            fila_entrada = int(tablas.cierres.index.searchsorted(pd.Timestamp(posicion["fecha_entrada"]), side="left"))
            if fila - fila_entrada >= p.max_dias_posicion:
                return "tiempo"
        if self.salida_personalizada(ticker, fecha, dict(posicion), tablas.cierres[ticker].iloc[: fila + 1]):
            return "regla_propia"
        return None

    def _aplicar_salidas(self, posiciones: dict[str, Any], enfriamiento: dict[str, str], tablas: _Tablas, fila: int,
                         fecha: datetime, universo: set[str], eventos: list[dict[str, Any]]) -> set[str]:
        """Cierra las posiciones que cumplen alguna salida; devuelve los tickers vendidos en esta llamada."""
        vendidos: set[str] = set()
        dia = f"{pd.Timestamp(fecha):%Y-%m-%d}"
        for ticker in sorted(posiciones):
            motivo = self._motivo_de_salida(ticker, posiciones[ticker], tablas, fila, fecha, universo)
            if motivo is None:
                continue
            precio = self._precio(tablas, ticker, fila)
            eventos.append({"fecha": dia, "ticker": ticker, "tipo": "salida", "motivo": motivo, "precio": precio})
            if motivo in MOTIVOS_CON_ENFRIAMIENTO and self.parametros.enfriamiento_dias > 0:
                fin = pd.Timestamp(fecha) + pd.offsets.BDay(self.parametros.enfriamiento_dias)
                enfriamiento[ticker] = f"{fin:%Y-%m-%d}"
            del posiciones[ticker]
            vendidos.add(ticker)
        return vendidos

    def _aplicar_entradas(self, posiciones: dict[str, Any], enfriamiento: dict[str, str], vendidos: set[str],
                          tablas: _Tablas, fila: int, fecha: datetime, universo: set[str],
                          eventos: list[dict[str, Any]]) -> None:
        """Abre posiciones sobre los huecos libres con los mejores candidatos y el tamaño de `ponderacion`."""
        p = self.parametros
        huecos = p.max_posiciones - len(posiciones)
        if huecos <= 0:
            return
        dia = f"{pd.Timestamp(fecha):%Y-%m-%d}"
        candidatos = [
            t for t in sorted(universo & set(tablas.cierres.columns))
            if t not in posiciones and t not in vendidos and enfriamiento.get(t, "") <= dia
            and bool(tablas.entrada[t].iloc[fila]) and not bool(tablas.salida[t].iloc[fila])
            and self._precio(tablas, t, fila) is not None
        ]
        if not candidatos:
            return
        if tablas.puntuacion is not None:
            valores = {t: tablas.puntuacion[t].iloc[fila] for t in candidatos}
            candidatos.sort(key=lambda t: (-valores[t] if math.isfinite(valores[t]) else math.inf, t))
        elegidos = candidatos[:huecos]
        pesos = self._tamanos(elegidos, tablas, fila)
        libre = max(1.0 - sum(pos["peso"] for pos in posiciones.values()), 0.0)
        total = sum(pesos.values())
        escala = libre / total if total > libre else 1.0
        for ticker in elegidos:
            precio = self._precio(tablas, ticker, fila)
            peso = pesos[ticker] * escala
            if peso <= UMBRAL_EFECTIVO:
                continue
            posiciones[ticker] = {"fecha_entrada": dia, "precio_entrada": precio, "maximo": precio, "peso": peso}
            eventos.append({"fecha": dia, "ticker": ticker, "tipo": "entrada", "motivo": "senal", "precio": precio})

    def _tamanos(self, elegidos: list[str], tablas: _Tablas, fila: int) -> dict[str, float]:
        """Peso de entrada de cada elegido: `1/max_posiciones` o repartido por 1/volatilidad, con el tope `peso_maximo`."""
        p = self.parametros
        base = 1.0 / p.max_posiciones
        relativos = {t: 1.0 for t in elegidos}
        if p.ponderacion == "inversa_volatilidad" and tablas.volatilidad is not None:
            inversas = {}
            for t in elegidos:
                vol = tablas.volatilidad[t].iloc[fila]
                if math.isfinite(vol) and vol > 0:
                    inversas[t] = 1.0 / vol
            if inversas:
                media = sum(inversas.values()) / len(inversas)
                relativos = {t: inversas.get(t, media) / media for t in elegidos}  # sin volatilidad medible: tamaño medio
        return {t: min(base * relativos[t], p.peso_maximo) for t in elegidos}

