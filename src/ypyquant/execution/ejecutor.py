"""
ejecutor.py — Ejecutor: de pesos objetivo a órdenes (módulo puro, sin I/O).

Responsabilidad:
    Única conversión pesos objetivo -> órdenes del sistema. Antes había cuatro con reglas distintas
    (motor de backtest, bot de Alpaca, estrategia de Nautilus, bot heredado de GrowSave). Interfaz:

        planificar_rebalanceo(pesos, precios, cuenta, politica, tickers_bloqueados) -> PlanRebalanceo

    - `EstadoCuenta`: equity, efectivo y posiciones (cantidad por ticker; NEGATIVA = corto) en el momento de decidir.
    - `PoliticaEjecucion`: lo que cambia de un consumidor a otro: acciones enteras o fracciones,
      umbral absoluto (divisa) y relativo (fracción del equity), si se normaliza una suma de pesos
      desviada y si se permiten cortos (`permite_cortos`). `POLITICA_MOTOR` (la del backtest, intacta) y `POLITICA_BOT` (Alpaca/Nautilus).
    - `PlanRebalanceo`: `ordenes` (ventas y cierres primero), `omitidas` (con motivo: bloqueado,
      sin_precio, bajo_umbral) y `rotacion` (valor a operar / 2·equity, comparado con la deriva).

    Semántica de CASH (`TICKER_EFECTIVO`): es el peso que se queda en efectivo. NUNCA genera orden ni se
    omite «por no tener precio»: se separa explícitamente antes de decidir. Su valor entra en la
    normalización de la suma (un 0.5 + CASH 0.5 suma 1) y el efectivo no invertido es la diferencia
    hasta el equity. La rotación no cuenta el CASH como valor operado: hasta 2026-11-22 el motor sí lo
    hacía (el 50 % en CASH daba una rotación fantasma de 0.5 en cada rebalanceo); los experimentos
    guardados antes de esa fecha con CASH tienen el turnover inflado (los costes no: no hubo órdenes).

    Cortos (solo con `politica.permite_cortos`; sin ella un peso <= 0 se ignora como siempre): un peso NEGATIVO
    pide una posición corta de `|peso| x equity`. Una venta sin posición larga abre o amplía el corto; una compra
    sobre un corto lo cubre. Si el signo del objetivo cambia (largo -> corto o al revés) se emiten DOS órdenes: el
    cierre de la posición actual (motivo 'cierre') y la apertura de la contraria, porque los brokers no cruzan por
    cero en una sola orden. Con cortos no se normaliza la suma de pesos (son exposiciones explícitas; lo no asignado
    queda en efectivo) y el CASH negativo sigue siendo un préstamo. Las reglas de margen (límite de exposición,
    mantenimiento) NO las decide el Ejecutor: las aplica el adaptador (`ypyquant.execution.margen`).

    Orden de las órdenes: primero los cierres (peso <= 0 o fuera del objetivo, en el orden de las
    posiciones), después el resto por (objetivo - actual) ascendente, es decir, ventas antes que compras
    y, dentro de cada lado, la diferencia menor primero (empates: orden de `pesos`). El motor lo necesita
    así para reproducir al céntimo qué compra se recorta si falta efectivo.

    El Ejecutor decide; no cobra comisiones ni aplica slippage: eso es del adaptador que ejecuta
    (el motor con su modelo de costes, el simulador o un broker real). Cada `Orden.importe` es el valor a
    precio de referencia que el adaptador debe comprar o vender.

Se relaciona con:
    - ypyquant.portfolio.backtest_engine: `_execute_rebalance` planifica con POLITICA_MOTOR y aplica
      las órdenes con `_buy`/`_sell` (modelo de costes propio).
    - pipelines/paper_trading/alpaca_vivo.py: planifica con POLITICA_BOT y envía por un `Broker`
      (ver broker.py).
    - ypyquant.execution.nautilus.ordenes: la decisión por señal (`decidir_orden`) es el otro camino de
      entrada del bot; ambos desembocan en el mismo envío.
    - ypyquant.execution.broker / broker_simulado: reciben las órdenes planificadas.
    - Paper interno (docs/planes/2026-11-20-configuraciones-y-paper-interno.md, fase 3): lo usa como núcleo.
"""

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TICKER_EFECTIVO = "CASH"

COMPRA = "compra"
VENTA = "venta"


@dataclass(frozen=True)
class EstadoCuenta:
    """Foto de la cuenta: `equity` total, `efectivo` y `posiciones` ({ticker: cantidad}, solo largos)."""
    equity: float
    efectivo: float
    posiciones: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PoliticaEjecucion:
    """
    Reglas de conversión que difieren entre consumidores.

    - fraccionable: True = cantidades fraccionarias (backtest); False = acciones enteras con floor (broker).
    - umbral_absoluto / umbral_relativo: no se opera una diferencia de valor menor que el mayor de
      `umbral_absoluto` (divisa) y `umbral_relativo * equity`; los cierres de posición no lo respetan.
    - tolerancia_suma: si no es None y la suma de pesos (CASH incluido) se desvía más que esto de 1,
      se normaliza a 1; None = no se normaliza (lo no asignado queda en efectivo).
    - permite_cortos: True = los pesos negativos (excepto CASH) son posiciones cortas; False = se ignoran (largo solo).
      Con True la suma de pesos no se normaliza.
    """
    fraccionable: bool = True
    umbral_absoluto: float = 0.0
    umbral_relativo: float = 0.0
    tolerancia_suma: float | None = None
    permite_cortos: bool = False


# Política del motor de backtest: fracciones, umbral de 10 unidades de divisa, normaliza si |suma-1| > 0.01
POLITICA_MOTOR = PoliticaEjecucion(fraccionable=True, umbral_absoluto=10.0, tolerancia_suma=0.01)
# Política del bot de Alpaca y de Nautilus: acciones enteras, umbral del 1 % del equity, sin normalizar
POLITICA_BOT = PoliticaEjecucion(fraccionable=False, umbral_relativo=0.01)


@dataclass(frozen=True)
class Orden:
    """
    Una orden de mercado: `lado` ('compra' | 'venta'), `cantidad` (entera si la política no es
    fraccionable), `importe` (valor a precio de referencia), `cantidad_objetivo` (posición final
    prevista, con signo: negativa = corto) y `motivo` ('cierre' si deja la posición a cero, 'rebalanceo' en otro caso).
    """
    ticker: str
    lado: str
    cantidad: float
    importe: float
    cantidad_objetivo: float
    motivo: str = "rebalanceo"


@dataclass(frozen=True)
class Omision:
    """Ticker sobre el que no se emite orden y por qué: 'bloqueado', 'sin_precio' o 'bajo_umbral'."""
    ticker: str
    motivo: str


@dataclass(frozen=True)
class PlanRebalanceo:
    """Resultado de planificar: órdenes en orden de envío, omisiones con motivo y rotación del rebalanceo."""
    ordenes: tuple[Orden, ...]
    omitidas: tuple[Omision, ...]
    rotacion: float


def _normalizar(pesos: Mapping[str, float], politica: PoliticaEjecucion) -> dict[str, float]:
    """Normaliza a suma 1 si la política lo pide y la suma se desvía más que la tolerancia."""
    pesos = dict(pesos)
    if politica.permite_cortos:
        return pesos  # exposiciones explícitas (largas y cortas): lo no asignado queda en efectivo
    total = sum(pesos.values())
    if politica.tolerancia_suma is not None and abs(total - 1.0) > politica.tolerancia_suma and total > 0:
        logger.warning(f"Pesos no suman 1.0 (suma: {total}), normalizando")
        return {t: w / total for t, w in pesos.items()}
    return pesos


def _rotacion(pesos: Mapping[str, float], precios: Mapping[str, float], cuenta: EstadoCuenta) -> float:
    """
    Valor que hay que operar (posiciones a la deriva -> objetivo, más el cambio de efectivo) entre
    2·equity. Un ticker sin precio se valora a 0 en la cartera actual (comportamiento histórico).
    """
    if cuenta.equity <= 0:
        return 0.0
    valores_objetivo = {t: cuenta.equity * w for t, w in pesos.items() if t != TICKER_EFECTIVO}
    valores_actuales = {t: q * precios.get(t, 0.0) for t, q in cuenta.posiciones.items()}
    operado = sum(abs(valores_objetivo.get(t, 0.0) - valores_actuales.get(t, 0.0))
                  for t in sorted(set(valores_objetivo) | set(valores_actuales)))
    efectivo_objetivo = cuenta.equity - sum(valores_objetivo.values())
    operado += abs(efectivo_objetivo - cuenta.efectivo)
    return float(operado / (2 * cuenta.equity))


def planificar_rebalanceo(
    pesos: Mapping[str, float],
    precios: Mapping[str, float],
    cuenta: EstadoCuenta,
    politica: PoliticaEjecucion,
    tickers_bloqueados: Collection[str] = (),
) -> PlanRebalanceo:
    """
    Órdenes para llevar `cuenta` a `pesos` con las reglas de `politica` (largo solo salvo `politica.permite_cortos`).

    `precios` son los de referencia de cada ticker; un ticker sin precio positivo no se opera (queda
    en `omitidas`). `tickers_bloqueados` (p. ej. con una orden aún abierta en el broker) no se toca ni
    para cerrarlo. CASH se interpreta como efectivo y nunca genera orden.
    """
    pesos = _normalizar(pesos, politica)
    rotacion = _rotacion(pesos, precios, cuenta)
    bloqueados = set(tickers_bloqueados)
    cortos = politica.permite_cortos
    posiciones = {t: q for t, q in cuenta.posiciones.items() if t != TICKER_EFECTIVO and (q > 0 or (cortos and q < 0))}
    omitidas: list[Omision] = []

    def _precio(ticker: str) -> float | None:
        valor = precios.get(ticker)
        return valor if valor is not None and valor > 0 else None

    def _cantidad(valor: float) -> float:
        return float(int(valor)) if not politica.fraccionable else valor

    def _mantiene(ticker: str, cantidad: float) -> bool:
        """True si el objetivo conserva el lado de la posición actual (largo con peso > 0, corto con peso < 0)."""
        peso = pesos.get(ticker, 0.0)
        return peso > 0 if cantidad > 0 else peso < 0

    # 1. Cierres: posiciones cuyo objetivo es 0 o de signo contrario, antes que nada y sin umbral
    #    (un corto se cierra con una compra; un cambio de signo se resuelve aquí y abre en el paso 2)
    ordenes: list[Orden] = []
    cerrados: set[str] = set()
    for ticker, cantidad in posiciones.items():
        if _mantiene(ticker, cantidad):
            continue
        if ticker in bloqueados:
            omitidas.append(Omision(ticker, "bloqueado"))
        elif (precio := _precio(ticker)) is None:
            omitidas.append(Omision(ticker, "sin_precio"))
        elif _cantidad(abs(cantidad)) > 0:
            lado = VENTA if cantidad > 0 else COMPRA
            ordenes.append(Orden(ticker, lado, _cantidad(abs(cantidad)), abs(cantidad) * precio, 0.0, "cierre"))
            cerrados.add(ticker)

    # 2. Resto de tickers con peso distinto de 0: diferencia entre objetivo y posición actual (0 si se acaba de cerrar)
    candidatas: list[tuple[float, Orden]] = []
    for ticker, peso in pesos.items():
        if ticker == TICKER_EFECTIVO or peso == 0 or (peso < 0 and not cortos):
            continue
        if ticker in bloqueados:
            omitidas.append(Omision(ticker, "bloqueado"))
            continue
        precio = _precio(ticker)
        if precio is None:
            omitidas.append(Omision(ticker, "sin_precio"))
            continue
        actual = 0.0 if ticker in cerrados else posiciones.get(ticker, 0.0)
        valor_actual = actual * precio
        valor_objetivo = cuenta.equity * peso
        if politica.fraccionable:
            diferencia = valor_objetivo - valor_actual
            cantidad_objetivo = valor_objetivo / precio
            cierra = False
            valor_operado = abs(diferencia)
            cantidad_orden = abs(diferencia) / precio
            importe = abs(diferencia)
        else:
            cantidad_objetivo = float(int(valor_objetivo / precio))
            actual_entera = float(int(actual))
            unidades = cantidad_objetivo - actual_entera
            diferencia = unidades * precio
            cierra = cantidad_objetivo == 0 and actual_entera != 0
            valor_operado = abs(unidades) * precio
            cantidad_orden = abs(unidades)
            importe = abs(unidades) * precio
        umbral = max(politica.umbral_absoluto, politica.umbral_relativo * cuenta.equity)
        sin_cambio = diferencia == 0 if politica.fraccionable else cantidad_orden == 0
        if sin_cambio:
            continue
        if valor_operado < umbral and not cierra:
            omitidas.append(Omision(ticker, "bajo_umbral"))
            continue
        lado = COMPRA if diferencia > 0 else VENTA
        if lado == VENTA and actual <= 0 and peso > 0:
            continue  # un objetivo largo nunca se alcanza vendiendo sin posición
        motivo = "cierre" if cierra else "rebalanceo"
        candidatas.append((diferencia, Orden(ticker, lado, cantidad_orden, importe, cantidad_objetivo, motivo)))

    candidatas.sort(key=lambda par: par[0])  # estable: empates en el orden de `pesos`
    ordenes.extend(orden for _, orden in candidatas)
    return PlanRebalanceo(tuple(ordenes), tuple(omitidas), rotacion)
