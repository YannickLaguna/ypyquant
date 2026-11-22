"""
Vocabulario de datos del marco de estrategias de señales: señales, gestión de la posición, límites y motivos.

Responsabilidad: definir, SIN lógica de mercado ni pandas, los objetos que se pasan entre las tres piezas del marco:
(1) las reglas del autor, que producen `Senal` (qué pasó: «AAPL pide entrar largo con fuerza 0,8»);
(2) la gestión de la posición (`GestionPosicion`: stops, take profit, trailing, tiempo, enfriamiento, giro), que decide
cuándo se sale; y (3) el dimensionamiento (`Dimensionamiento` y `LimitesExposicion`), que decide cuánto se entra y
cuánta exposición bruta/neta se tolera. Separarlas permite cambiar el tamaño sin tocar las reglas y viceversa.

Contenido:
- `LARGO` / `CORTO`: los dos lados. Una posición en dict JSON lleva `lado`, `fecha_entrada`, `precio_entrada`, `peso`
  (positivo en largos, NEGATIVO en cortos) y su extremo favorable (`maximo` en largos, `minimo` en cortos).
- `Senal(ticker, lado, tipo, fuerza)`: dato inmutable; `tipo` es `entrada` o `salida`; `fuerza` ordena candidatos y puede
  dimensionar (None = sin puntuación).
- Motivos de cada operación (`ENTRADA_LARGA`, `STOP_LOSS`...) y `a_motivo_actual` / `a_motivo_legado`, que traducen entre
  el vocabulario actual y el anterior (`senal`, `stop_perdida`...) para leer experimentos guardados antes del rediseño.
- `nivel_stop`, `nivel_take_profit`, `nivel_trailing`: precio de cada nivel según el lado (en un corto el stop está ENCIMA).

Se relaciona con: .senales_maquina (consume estas clases), .senales_tamano (dimensionamiento), .senales (adaptador de
pandas y parámetros), .simulacion y .evaluacion (usan los motivos) y docs/estrategias_senales.md (guía).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Lado = Literal["largo", "corto"]
TipoSenal = Literal["entrada", "salida"]

LARGO: Lado = "largo"
CORTO: Lado = "corto"

# Motivos de cada operación (columna `motivo` de las operaciones y de los eventos de la estrategia)
ENTRADA_LARGA = "entrada_larga"
ENTRADA_CORTA = "entrada_corta"
STOP_LOSS = "stop_loss"
TAKE_PROFIT = "take_profit"
TRAILING = "trailing"
TIEMPO = "tiempo"
SENAL_SALIDA = "senal_salida"          # la regla de salida del lado de la posición
SENAL_CONTRARIA = "senal_contraria"    # entra la señal del lado opuesto: se cierra (y, con giro, se abre la contraria)
SALIDA_UNIVERSO = "salida_universo"
REGLA_PROPIA = "regla_propia"
LLAMADA_MARGEN = "llamada_margen"      # liquidación forzosa del motor por falta de margen
AJUSTE = "ajuste"                      # compra o venta parcial que solo recoloca pesos (deriva, costes)

MOTIVOS_ENTRADA = (ENTRADA_LARGA, ENTRADA_CORTA)
MOTIVOS_CON_ENFRIAMIENTO = (STOP_LOSS, TRAILING)  # tras una salida de pérdida se espera antes de reentrar
MOTIVOS = (ENTRADA_LARGA, ENTRADA_CORTA, STOP_LOSS, TAKE_PROFIT, TRAILING, TIEMPO, SENAL_SALIDA, SENAL_CONTRARIA,
           SALIDA_UNIVERSO, REGLA_PROPIA, LLAMADA_MARGEN, AJUSTE)

_LEGADO_SALIDAS = {"senal": SENAL_SALIDA, "stop_perdida": STOP_LOSS, "stop_trailing": TRAILING,
                   "fuera_universo": SALIDA_UNIVERSO}


def a_motivo_actual(motivo: str | None, es_entrada: bool) -> str | None:
    """
    Traduce un motivo del vocabulario anterior al actual (`senal` de una entrada -> `entrada_larga`; `senal` de una
    salida -> `senal_salida`; `stop_perdida` -> `stop_loss`...). Los motivos ya actuales y `None` pasan tal cual.
    Sirve para leer operaciones guardadas antes de que el marco tuviera cortos (todas eran largas).
    """
    if motivo is None:
        return None
    if motivo in ("senal", "entrada") and es_entrada:
        return ENTRADA_LARGA
    return _LEGADO_SALIDAS.get(motivo, motivo)


def a_motivo_legado(motivo: str) -> str:
    """Inversa de `a_motivo_actual` para estrategias solo largas: `entrada_larga` y `senal_salida` -> `senal`, etc."""
    inversos = {v: k for k, v in _LEGADO_SALIDAS.items() if k != "senal"}
    if motivo in (ENTRADA_LARGA, SENAL_SALIDA):
        return "senal"
    return inversos.get(motivo, motivo)


@dataclass(frozen=True)
class Senal:
    """
    Una señal como dato: el ticker pide `entrada` o `salida` en un `lado`.

    `fuerza` (opcional) es la puntuación de la señal: ordena a los candidatos cuando hay más que huecos (mayor = antes) y,
    con `ponderacion="fuerza"`, reparte el tamaño. None = sin puntuación (orden alfabético, tamaño igual).
    """

    ticker: str
    lado: Lado
    tipo: TipoSenal
    fuerza: float | None = None

    def __post_init__(self) -> None:
        if self.lado not in (LARGO, CORTO):
            raise ValueError(f"lado inválido: {self.lado!r} (largo | corto)")
        if self.tipo not in ("entrada", "salida"):
            raise ValueError(f"tipo inválido: {self.tipo!r} (entrada | salida)")


@dataclass(frozen=True)
class GestionPosicion:
    """
    Cómo se gestiona una posición ya abierta (el mismo parámetro vale para largos y cortos, con el signo adecuado).

    stop_perdida: fracción de pérdida desde la entrada (largo: el precio cae; corto: el precio SUBE).
    take_profit: fracción de ganancia desde la entrada (largo: sube; corto: cae).
    stop_trailing: fracción que el precio retrocede desde su extremo favorable (máximo en largos, mínimo en cortos).
    max_dias_posicion: sesiones máximas en cartera. enfriamiento_dias: días hábiles sin reentrar tras stop_loss o trailing.
    permitir_giro: con señal contraria, cierra y abre la posición opuesta el MISMO día (sin giro, queda plano esa sesión).
    mantener_deriva: los pesos derivan con el precio entre llamadas (sin operar si no hay cambios).
    lados: lados que la estrategia puede abrir.
    """

    stop_perdida: float | None = None
    take_profit: float | None = None
    stop_trailing: float | None = None
    max_dias_posicion: int | None = None
    enfriamiento_dias: int = 5
    permitir_giro: bool = False
    mantener_deriva: bool = True
    lados: tuple[Lado, ...] = (LARGO,)


@dataclass(frozen=True)
class LimitesExposicion:
    """
    Cuántas posiciones y cuánta exposición se toleran (fracciones del equity).

    max_posiciones: posiciones simultáneas totales; max_largas / max_cortas: tope por lado (None = solo el total).
    exposicion_bruta_max: suma de |pesos| máxima. exposicion_neta_max / exposicion_neta_min: suma de pesos (largos menos
    cortos) en [min, max]. Con los valores por defecto (1, 1, -1) y solo largos, la cartera nunca supera el capital.
    """

    max_posiciones: int = 10
    max_largas: int | None = None
    max_cortas: int | None = None
    exposicion_bruta_max: float = 1.0
    exposicion_neta_max: float = 1.0
    exposicion_neta_min: float = -1.0

    @property
    def admite_prestamo(self) -> bool:
        """True si la exposición neta puede pasar de 1 (el efectivo queda negativo: préstamo de margen)."""
        return self.exposicion_neta_max > 1.0


@dataclass(frozen=True)
class Dimensionamiento:
    """
    Cómo se reparte el tamaño de las entradas (peso por posición, siempre como magnitud positiva).

    ponderacion: `iguales` (1/max_posiciones), `inversa_volatilidad`, `fuerza` (proporcional a la fuerza de la señal) o
    `riesgo` (el peso es `riesgo_por_operacion / stop_perdida`: lo que se pierde al saltar el stop es `riesgo_por_operacion`
    del equity). peso_maximo: tope por posición.
    """

    ponderacion: str = "iguales"
    max_posiciones: int = 10
    peso_maximo: float = 0.25
    riesgo_por_operacion: float = 0.01
    stop_perdida: float | None = None


def es_largo(posicion: dict[str, Any]) -> bool:
    """Lado de una posición en dict; las guardadas antes de los cortos no traen `lado` y son largas."""
    return posicion.get("lado", LARGO) == LARGO


def nivel_stop(posicion: dict[str, Any], stop_perdida: float) -> float:
    """Precio del stop de pérdida: DEBAJO de la entrada en un largo, ENCIMA en un corto."""
    entrada = posicion["precio_entrada"]
    return entrada * (1.0 - stop_perdida) if es_largo(posicion) else entrada * (1.0 + stop_perdida)


def nivel_take_profit(posicion: dict[str, Any], take_profit: float) -> float:
    """Precio del take profit: ENCIMA de la entrada en un largo, DEBAJO en un corto."""
    entrada = posicion["precio_entrada"]
    return entrada * (1.0 + take_profit) if es_largo(posicion) else entrada * (1.0 - take_profit)


def nivel_trailing(posicion: dict[str, Any], stop_trailing: float) -> float:
    """Precio del trailing: retroceso desde el máximo (largo: por debajo) o desde el mínimo (corto: por encima)."""
    if es_largo(posicion):
        return posicion["maximo"] * (1.0 - stop_trailing)
    return posicion["minimo"] * (1.0 + stop_trailing)
