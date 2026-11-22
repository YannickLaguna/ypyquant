"""
Máquina de estados por ticker del marco de señales: plano -> largo / corto, con gestión de la posición y límites.

Responsabilidad: dado el estado (posiciones abiertas y enfriamiento, JSON-serializables), las señales del día
(`Senal`) y una vista del mercado en la fecha D (`Mercado`), decidir salidas y entradas y devolver los eventos con su
motivo. No conoce pandas, tablas ni indicadores: solo señales como datos y precios por callable. Eso la hace testeable
con casos construidos a mano y con Hypothesis (invariantes: nunca largo y corto a la vez en un ticker, el stop siempre
del lado correcto, la exposición dentro de los límites).

Modelo (en cada llamada, en este orden, como el marco anterior):
1. Enfriamientos vencidos se descartan. 2. Deriva: los pesos cambian con el precio desde la llamada anterior y se
actualiza el extremo favorable (máximo en largos, mínimo en cortos). 3. Salidas, por prioridad: fuera del universo,
stop_loss, trailing, take_profit, señal contraria, señal de salida, tiempo, regla propia. 4. Entradas largas y luego cortas
sobre los huecos libres, ordenadas por fuerza, dimensionadas (`senales_tamano`) y escaladas para no pasar de los límites
de exposición bruta y neta.

Simetría de los cortos: el stop de pérdida está ENCIMA de la entrada (el precio sube), el take profit DEBAJO, el trailing
sigue al MÍNIMO. El peso de un corto es negativo. Un ticker con entrada larga y corta a la vez no entra por ningún lado
(señales contradictorias). Una entrada y una salida del mismo lado el mismo día tampoco entra (como el marco anterior).

Decisión sin look-ahead: la máquina solo ve el estado y las señales de la fecha D; que esas señales sean causales es
responsabilidad de las reglas (`verificar_sin_lookahead`). La ejecución a cierre o apertura siguiente la decide el motor.

Se relaciona con: .senales_modelo (datos), .senales_tamano (tamaños), .senales (adaptador de pandas: construye `Mercado`
y las señales, y devuelve los pesos al motor) y tests/test_senales_maquina.py (propiedades con Hypothesis).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ypyquant.estrategias.senales_modelo import (
    CORTO,
    ENTRADA_CORTA,
    ENTRADA_LARGA,
    LARGO,
    MOTIVOS_CON_ENFRIAMIENTO,
    REGLA_PROPIA,
    SALIDA_UNIVERSO,
    SENAL_CONTRARIA,
    SENAL_SALIDA,
    STOP_LOSS,
    TAKE_PROFIT,
    TIEMPO,
    TRAILING,
    Dimensionamiento,
    GestionPosicion,
    LimitesExposicion,
    Senal,
    es_largo,
)
from ypyquant.estrategias.senales_tamano import tamanos_de_entrada

UMBRAL_EFECTIVO = 1e-9  # por debajo de este peso no se abre la posición


@dataclass
class Mercado:
    """
    Lo que la máquina ve del mercado en la fecha D (todo con datos hasta D).

    fecha: `YYYY-MM-DD`. universo: tickers elegibles hoy. hay_previa: hubo una llamada anterior (hay deriva que aplicar).
    precio(t): cierre de D (None si no hay o no es válido). crecimiento(t): precio de D / precio de la llamada anterior
    (1.0 si no se puede calcular). extremos_tramo(t): `(máximo, mínimo)` de los cierres desde la llamada anterior (None
    si no hay). sesiones_desde(fecha_entrada): sesiones transcurridas desde esa fecha. volatilidad(t): volatilidad reciente
    para el dimensionamiento. regla_propia(t, posicion): salida a medida del autor (False si no la define).
    """

    fecha: str
    universo: frozenset[str] | set[str]
    precio: Callable[[str], float | None]
    hay_previa: bool = False
    crecimiento: Callable[[str], float] = lambda ticker: 1.0
    extremos_tramo: Callable[[str], tuple[float, float] | None] = lambda ticker: None
    sesiones_desde: Callable[[str], int] = lambda fecha_entrada: 0
    volatilidad: Callable[[str], float | None] = lambda ticker: None
    regla_propia: Callable[[str, dict[str, Any]], bool] = lambda ticker, posicion: False


@dataclass
class _SenalesDelDia:
    """Señales de la fecha agrupadas por lado: entradas `{ticker: fuerza}` y salidas `{ticker}`."""

    entradas: dict[str, dict[str, float | None]] = field(default_factory=lambda: {LARGO: {}, CORTO: {}})
    salidas: dict[str, set[str]] = field(default_factory=lambda: {LARGO: set(), CORTO: set()})

    @classmethod
    def de(cls, senales: Iterable[Senal]) -> _SenalesDelDia:
        agrupadas = cls()
        for senal in senales:
            if senal.tipo == "entrada":
                agrupadas.entradas[senal.lado][senal.ticker] = senal.fuerza
            else:
                agrupadas.salidas[senal.lado].add(senal.ticker)
        return agrupadas

    def entrada_valida(self, lado: str, ticker: str) -> bool:
        """Entrada del lado sin salida del mismo lado ni entrada contradictoria del lado opuesto."""
        otro = CORTO if lado == LARGO else LARGO
        return ticker in self.entradas[lado] and ticker not in self.salidas[lado] and ticker not in self.entradas[otro]


def lado_opuesto(lado: str) -> str:
    """El lado contrario."""
    return CORTO if lado == LARGO else LARGO


class MaquinaPosiciones:
    """
    Máquina de estados de las posiciones de una estrategia de señales (ver cabecera del módulo).

    Se construye con la gestión (`GestionPosicion`), los límites (`LimitesExposicion`) y el dimensionamiento
    (`Dimensionamiento`); no guarda estado propio: `avanzar` modifica el dict de posiciones y el de enfriamiento que
    recibe (los que persiste el paper y el bot), y devuelve los eventos de la llamada.
    """

    def __init__(self, gestion: GestionPosicion, limites: LimitesExposicion, dimension: Dimensionamiento):
        self.gestion = gestion
        self.limites = limites
        self.dimension = dimension

    # ---- API ---------------------------------------------------------------------------------------------

    def avanzar(self, posiciones: dict[str, dict[str, Any]], enfriamiento: dict[str, str], senales: Iterable[Senal],
                mercado: Mercado) -> list[dict[str, Any]]:
        """
        Aplica un día: deriva, salidas y entradas sobre `posiciones` y `enfriamiento` (se modifican) y devuelve los
        eventos `{fecha, ticker, tipo: 'entrada'|'salida', motivo, precio, lado}` en orden.
        """
        for ticker in [t for t, fin in enfriamiento.items() if fin <= mercado.fecha]:
            del enfriamiento[ticker]
        agrupadas = _SenalesDelDia.de(senales)
        eventos: list[dict[str, Any]] = []
        self._aplicar_deriva(posiciones, mercado)
        vendidos, giros = self._aplicar_salidas(posiciones, enfriamiento, agrupadas, mercado, eventos)
        for lado in (LARGO, CORTO):
            if lado in self.gestion.lados:
                self._aplicar_entradas(lado, posiciones, enfriamiento, vendidos, giros, agrupadas, mercado, eventos)
        return eventos

    @staticmethod
    def pesos(posiciones: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Pesos con signo de las posiciones (largos positivos, cortos negativos), ordenados por ticker."""
        return {t: p["peso"] for t, p in sorted(posiciones.items())}

    # ---- deriva -------------------------------------------------------------------------------------------

    def _aplicar_deriva(self, posiciones: dict[str, dict[str, Any]], mercado: Mercado) -> None:
        """Actualiza el extremo favorable de cada posición y, con `mantener_deriva`, sus pesos según el precio."""
        if not mercado.hay_previa or not posiciones:
            return
        crecimiento = {}
        for ticker, posicion in posiciones.items():
            extremos = mercado.extremos_tramo(ticker)
            if extremos is not None:
                if es_largo(posicion):
                    posicion["maximo"] = max(posicion["maximo"], extremos[0])
                else:
                    posicion["minimo"] = min(posicion["minimo"], extremos[1])
            crecimiento[ticker] = mercado.crecimiento(ticker)
        if not self.gestion.mantener_deriva:
            return
        efectivo = 1.0 - sum(p["peso"] for p in posiciones.values())
        if efectivo < 0 and not self.limites.admite_prestamo:
            efectivo = 0.0  # solo largos sin apalancamiento: nunca hay préstamo (ruido de redondeo)
        total = max(efectivo + sum(p["peso"] * crecimiento[t] for t, p in posiciones.items()), UMBRAL_EFECTIVO)
        for ticker, posicion in posiciones.items():
            posicion["peso"] = posicion["peso"] * crecimiento[ticker] / total

    # ---- salidas ------------------------------------------------------------------------------------------

    def _motivo_de_salida(self, ticker: str, posicion: dict[str, Any], agrupadas: _SenalesDelDia,
                          mercado: Mercado) -> str | None:
        """Primer motivo por el que la posición debe cerrarse (ver orden en la cabecera); None si se mantiene."""
        g = self.gestion
        largo = es_largo(posicion)
        lado = LARGO if largo else CORTO
        precio = mercado.precio(ticker)
        if ticker not in mercado.universo:
            return SALIDA_UNIVERSO
        if precio is not None:
            rendimiento = precio / posicion["precio_entrada"] - 1.0  # variación del precio; en un corto, perder = subir
            if g.stop_perdida is not None and (rendimiento <= -g.stop_perdida if largo else rendimiento >= g.stop_perdida):
                return STOP_LOSS
            if g.stop_trailing is not None:
                if largo and precio / posicion["maximo"] - 1.0 <= -g.stop_trailing:
                    return TRAILING
                if not largo and precio / posicion["minimo"] - 1.0 >= g.stop_trailing:
                    return TRAILING
            if g.take_profit is not None and (rendimiento >= g.take_profit if largo else rendimiento <= -g.take_profit):
                return TAKE_PROFIT
        otro = lado_opuesto(lado)
        if otro in g.lados and agrupadas.entrada_valida(otro, ticker):
            return SENAL_CONTRARIA
        if ticker in agrupadas.salidas[lado]:
            return SENAL_SALIDA
        if g.max_dias_posicion is not None and mercado.sesiones_desde(posicion["fecha_entrada"]) >= g.max_dias_posicion:
            return TIEMPO
        if mercado.regla_propia(ticker, dict(posicion)):
            return REGLA_PROPIA
        return None

    def _aplicar_salidas(self, posiciones: dict[str, dict[str, Any]], enfriamiento: dict[str, str], agrupadas: _SenalesDelDia,
                         mercado: Mercado, eventos: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
        """Cierra las posiciones que cumplen alguna salida; devuelve (tickers cerrados hoy, los que pueden girar hoy)."""
        vendidos: set[str] = set()
        giros: set[str] = set()
        for ticker in sorted(posiciones):
            posicion = posiciones[ticker]
            motivo = self._motivo_de_salida(ticker, posicion, agrupadas, mercado)
            if motivo is None:
                continue
            eventos.append({"fecha": mercado.fecha, "ticker": ticker, "tipo": "salida", "motivo": motivo,
                            "precio": mercado.precio(ticker), "lado": LARGO if es_largo(posicion) else CORTO})
            if motivo in MOTIVOS_CON_ENFRIAMIENTO and self.gestion.enfriamiento_dias > 0:
                fin = pd.Timestamp(mercado.fecha) + pd.offsets.BDay(self.gestion.enfriamiento_dias)
                enfriamiento[ticker] = f"{fin:%Y-%m-%d}"
            if motivo == SENAL_CONTRARIA and self.gestion.permitir_giro:
                giros.add(ticker)
            del posiciones[ticker]
            vendidos.add(ticker)
        return vendidos, giros

    # ---- entradas -----------------------------------------------------------------------------------------

    def _huecos(self, lado: str, posiciones: dict[str, dict[str, Any]]) -> int:
        """Posiciones que aún caben: en total y en el lado."""
        huecos = self.limites.max_posiciones - len(posiciones)
        tope = self.limites.max_largas if lado == LARGO else self.limites.max_cortas
        if tope is not None:
            huecos = min(huecos, tope - sum(1 for p in posiciones.values() if es_largo(p) == (lado == LARGO)))
        return huecos

    def _capacidad(self, lado: str, posiciones: dict[str, dict[str, Any]]) -> float:
        """Peso (magnitud) que aún se puede añadir en el lado sin pasar de la exposición bruta ni de la neta."""
        bruta = sum(abs(p["peso"]) for p in posiciones.values())
        neta = sum(p["peso"] for p in posiciones.values())
        holgura_neta = self.limites.exposicion_neta_max - neta if lado == LARGO else neta - self.limites.exposicion_neta_min
        return max(min(self.limites.exposicion_bruta_max - bruta, holgura_neta), 0.0)

    def _aplicar_entradas(self, lado: str, posiciones: dict[str, dict[str, Any]], enfriamiento: dict[str, str],
                          vendidos: set[str], giros: set[str], agrupadas: _SenalesDelDia, mercado: Mercado,
                          eventos: list[dict[str, Any]]) -> None:
        """Abre posiciones del lado sobre los huecos libres con los mejores candidatos y el tamaño de `dimension`."""
        huecos = self._huecos(lado, posiciones)
        if huecos <= 0:
            return
        candidatos = [
            t for t in sorted(agrupadas.entradas[lado])
            if t in mercado.universo and t not in posiciones and (t not in vendidos or t in giros)
            and enfriamiento.get(t, "") <= mercado.fecha and agrupadas.entrada_valida(lado, t)
            and mercado.precio(t) is not None
        ]
        if not candidatos:
            return
        fuerzas = agrupadas.entradas[lado]
        if any(fuerzas[t] is not None for t in candidatos):
            candidatos.sort(key=lambda t: (-fuerzas[t] if fuerzas[t] is not None and math.isfinite(fuerzas[t]) else math.inf, t))
        elegidos = candidatos[:huecos]
        pesos = tamanos_de_entrada(elegidos, fuerzas, mercado.volatilidad, self.dimension)
        libre = self._capacidad(lado, posiciones)
        total = sum(pesos.values())
        escala = libre / total if total > libre else 1.0
        for ticker in elegidos:
            precio = mercado.precio(ticker)
            peso = pesos[ticker] * escala
            if peso <= UMBRAL_EFECTIVO:
                continue
            if lado == LARGO:
                posiciones[ticker] = {"lado": LARGO, "fecha_entrada": mercado.fecha, "precio_entrada": precio,
                                      "maximo": precio, "peso": peso}
            else:
                posiciones[ticker] = {"lado": CORTO, "fecha_entrada": mercado.fecha, "precio_entrada": precio,
                                      "minimo": precio, "peso": -peso}
            eventos.append({"fecha": mercado.fecha, "ticker": ticker, "tipo": "entrada",
                            "motivo": ENTRADA_LARGA if lado == LARGO else ENTRADA_CORTA, "precio": precio, "lado": lado})
