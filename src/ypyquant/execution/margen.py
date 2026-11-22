"""
margen.py — Reglas de margen compartidas: requisito inicial, mantenimiento, llamadas de margen y costes (módulo puro).

Responsabilidad:
    UNA sola implementación de las reglas de margen y de cortos para todos los adaptadores que ejecutan: el motor de
    backtest (`BacktestEngine`), el simulador del paper interno (vía `BrokerSimulado`) y, como referencia de qué espera
    Alpaca, el vínculo de cuentas. Nada de I/O ni de estado: funciones puras sobre números.

    - `ReglasMargen` (dataclass inmutable): `apalancamiento_maximo` (exposición bruta máxima / equity), `requisito_inicial`
      (Reg T: 50 %), `mantenimiento_largo` (25 %) y `mantenimiento_corto` (30 %), `tasa_financiacion` y
      `coste_prestamo_titulos` (anuales), `permite_cortos`. Con los valores por defecto (L = 1, sin cortos) NO está
      `activo`: los adaptadores no entran en ninguna rama de margen y todo funciona como antes.
    - Definiciones. Exposición bruta = valor de largos + valor de cortos (valor absoluto). Equity = efectivo + largos -
      cortos (el efectivo incluye el producto de las ventas en corto). Requisito inicial: `equity >= requisito_inicial x
      bruta`, es decir bruta <= equity / 0,5 = 2x (límite Reg T overnight; `limite_bruta` = min(L, 1 / requisito_inicial)).
      Mantenimiento: `equity >= mantenimiento_largo x largos + mantenimiento_corto x cortos`.
    - `evaluar_margen`: estado diario (bruta, neta, requisito, `en_llamada` si el equity cae bajo el mantenimiento).
    - `factor_de_conservacion`: qué fracción de las posiciones se conserva en una liquidación forzosa para volver al
      requisito inicial, descontando el coste (comisión + slippage) de lo vendido. Se reduce TODO proporcionalmente.
    - `capacidad_de_compra` / `capacidad_bruta`: cuánto se puede añadir sin pasar del límite de exposición.
    - `politica_para`: la `PoliticaEjecucion` del Ejecutor que corresponde a las reglas (cortos sí/no).
    - `cargos_diarios`: interés del préstamo de margen (sobre el efectivo NEGATIVO, saldo x tasa x días / 360, convención de
      Alpaca) y coste de préstamo de títulos de los cortos (valor x tasa x días / 360).

Se relaciona con:
    - ypyquant.portfolio.backtest_engine: aplica estas reglas día a día (cargos, llamada de margen, capacidad de compra).
    - ypyquant.execution.broker_simulado: capacidad al comprar o vender en corto (paper interno).
    - ypyquant.paper.simulador: cargos diarios y llamada de margen de la sesión con las mismas funciones que el motor.
    - ypyquant.estrategias.simulacion: construye `ReglasMargen` desde los parámetros de la estrategia (`reglas_desde_parametros`).
    - ypyquant.perfiles_alpaca.vinculo: qué recetas admite una cuenta de margen.
    - docs/planes/2026-11-22-apalancamiento-y-cortos.md: estudio y decisiones.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ypyquant.execution.ejecutor import POLITICA_MOTOR, PoliticaEjecucion

DIAS_BASE_INTERES = 360.0  # convención de Alpaca: saldo x tasa / 360


@dataclass(frozen=True)
class ReglasMargen:
    """Reglas de margen de una cuenta (ver cabecera). Todos los porcentajes son fracciones (0,5 = 50 %)."""

    apalancamiento_maximo: float = 1.0
    requisito_inicial: float = 0.5
    mantenimiento_largo: float = 0.25
    mantenimiento_corto: float = 0.30
    tasa_financiacion: float = 0.0
    coste_prestamo_titulos: float = 0.0
    permite_cortos: bool = False

    def __post_init__(self) -> None:
        if self.apalancamiento_maximo < 1.0:
            raise ValueError("apalancamiento_maximo debe ser >= 1 (1 = sin apalancamiento)")
        if not 0.0 < self.requisito_inicial <= 1.0:
            raise ValueError("requisito_inicial debe estar en (0, 1]")
        if not 0.0 <= self.mantenimiento_largo < self.requisito_inicial or not 0.0 <= self.mantenimiento_corto < self.requisito_inicial:
            raise ValueError("los márgenes de mantenimiento deben estar en [0, requisito_inicial)")

    @property
    def activo(self) -> bool:
        """True si hay apalancamiento o cortos; si no, los adaptadores no aplican ninguna regla de margen."""
        return self.apalancamiento_maximo > 1.0 or self.permite_cortos

    @property
    def limite_bruta(self) -> float:
        """Exposición bruta máxima / equity: el apalancamiento pedido, acotado por el requisito inicial (Reg T: 2x)."""
        return min(self.apalancamiento_maximo, 1.0 / self.requisito_inicial)


@dataclass(frozen=True)
class EstadoMargen:
    """Foto del margen: `bruta` y `neta` (valores), `requisito_mantenimiento` y si el equity ya está por debajo (`en_llamada`)."""

    equity: float
    valor_largo: float
    valor_corto: float
    bruta: float
    neta: float
    requisito_mantenimiento: float
    en_llamada: bool

    @property
    def bruta_sobre_equity(self) -> float:
        if self.equity > 0:
            return self.bruta / self.equity
        return float("inf") if self.bruta > 0 else 0.0


def valores_largo_corto(posiciones: Mapping[str, float], precios: Mapping[str, float]) -> tuple[float, float]:
    """`(valor de largos, valor absoluto de cortos)` a los `precios` dados; un ticker sin precio no cuenta."""
    largo = corto = 0.0
    for ticker, cantidad in posiciones.items():
        precio = precios.get(ticker)
        if precio is None:
            continue
        if cantidad > 0:
            largo += cantidad * precio
        elif cantidad < 0:
            corto += -cantidad * precio
    return largo, corto


def evaluar_margen(equity: float, valor_largo: float, valor_corto: float, reglas: ReglasMargen) -> EstadoMargen:
    """
    Estado de margen de una cartera. `en_llamada` si hay posiciones y el equity cae por debajo del mantenimiento
    (`mantenimiento_largo x largos + mantenimiento_corto x cortos`); con las reglas inactivas nunca hay llamada.
    """
    requisito = reglas.mantenimiento_largo * valor_largo + reglas.mantenimiento_corto * valor_corto
    bruta = valor_largo + valor_corto
    en_llamada = reglas.activo and bruta > 0 and equity < requisito
    return EstadoMargen(equity, valor_largo, valor_corto, bruta, valor_largo - valor_corto, requisito, en_llamada)


def factor_de_conservacion(equity: float, bruta: float, reglas: ReglasMargen, coste_unitario: float = 0.0) -> float:
    """
    Fracción f en [0, 1] de las posiciones que se CONSERVA tras una liquidación forzosa (se vende `1 - f` de cada una).

    Se busca el mayor f con `equity - coste x (1 - f) x bruta >= requisito_inicial x f x bruta`, donde `coste_unitario` es
    comisión + slippage por unidad vendida: f = (equity - c x bruta) / (bruta x (r - c)). Con equity <= c x bruta se
    liquida todo (f = 0). Sin posiciones (bruta <= 0) no hay nada que reducir (f = 1).
    """
    if bruta <= 0:
        return 1.0
    r = reglas.requisito_inicial
    if equity <= 0 or r <= coste_unitario:
        return 0.0
    f = (equity - coste_unitario * bruta) / (bruta * (r - coste_unitario))
    return max(0.0, min(1.0, f))


def capacidad_bruta(equity: float, valor_largo: float, valor_corto: float, reglas: ReglasMargen) -> float:
    """Importe de exposición bruta que aún se puede añadir sin pasar de `limite_bruta x equity` (nunca negativo)."""
    return max(0.0, reglas.limite_bruta * equity - (valor_largo + valor_corto))


def capacidad_de_compra(efectivo: float, equity: float, valor_largo: float, valor_corto: float, reglas: ReglasMargen) -> float:
    """
    Importe máximo que se puede gastar en una compra (antes de comisión).

    Sin reglas activas es el efectivo. Con apalancamiento y sin cortos es `efectivo + (L - 1) x equity` (la fórmula de
    siempre del motor). Con cortos el efectivo incluye el producto de las ventas, así que manda el límite de exposición
    bruta (`capacidad_bruta`).
    """
    if not reglas.activo:
        return efectivo
    if valor_corto == 0.0 and not reglas.permite_cortos:
        return efectivo + (reglas.apalancamiento_maximo - 1.0) * equity
    return capacidad_bruta(equity, valor_largo, valor_corto, reglas)


def cargos_diarios(efectivo: float, valor_corto: float, dias: int, reglas: ReglasMargen) -> tuple[float, float]:
    """
    `(interés de margen, coste de préstamo de títulos)` de `dias` días naturales (el fin de semana cuenta 3).

    Interés: efectivo NEGATIVO x `tasa_financiacion` / 360. Préstamo de títulos: valor de cortos x
    `coste_prestamo_titulos` / 360. Ambos son 0 si el saldo/tasa son 0.
    """
    if dias <= 0:
        return 0.0, 0.0
    interes = -efectivo * reglas.tasa_financiacion * dias / DIAS_BASE_INTERES if efectivo < 0 and reglas.tasa_financiacion > 0 else 0.0
    prestamo = valor_corto * reglas.coste_prestamo_titulos * dias / DIAS_BASE_INTERES if valor_corto > 0 and reglas.coste_prestamo_titulos > 0 else 0.0
    return interes, prestamo


def politica_para(reglas: ReglasMargen, base: PoliticaEjecucion = POLITICA_MOTOR) -> PoliticaEjecucion:
    """
    Política del Ejecutor para unas reglas: `base` tal cual (la del motor por defecto) o, si las reglas permiten
    cortos, la misma con `permite_cortos=True` y sin normalizar la suma (las exposiciones son explícitas).
    """
    return replace(base, permite_cortos=True, tolerancia_suma=None) if reglas.permite_cortos else base


def reglas_desde_parametros(parametros: object, tasa_libre_riesgo: float | None = None) -> ReglasMargen:
    """
    `ReglasMargen` a partir de los parámetros de una estrategia (objeto Pydantic o dict con claves opcionales).

    Claves: `apalancamiento_maximo`, `diferencial_financiacion` (la tasa de financiación es `tasa_libre_riesgo` +
    diferencial, solo con apalancamiento), `exposicion_larga` y `exposicion_corta` (con `exposicion_corta` > 0 se permiten
    cortos y el límite de exposición bruta sube a `exposicion_larga + exposicion_corta` si es mayor) y
    `coste_prestamo_titulos`. Sin ninguna de ellas devuelve las reglas inactivas por defecto.
    """
    def obtener(nombre: str, defecto: float) -> float:
        valor = parametros.get(nombre) if isinstance(parametros, Mapping) else getattr(parametros, nombre, None)
        return float(defecto if valor is None else valor)

    apalancamiento = obtener("apalancamiento_maximo", 1.0)
    exposicion_corta = obtener("exposicion_corta", 0.0)
    cortos = exposicion_corta > 0.0
    if cortos:
        apalancamiento = max(apalancamiento, obtener("exposicion_larga", 1.0) + exposicion_corta)
    libre = obtener("tasa_libre_riesgo", 0.0) if tasa_libre_riesgo is None else tasa_libre_riesgo
    financiacion = libre + obtener("diferencial_financiacion", 0.0) if apalancamiento > 1.0 else 0.0
    return ReglasMargen(apalancamiento_maximo=apalancamiento, tasa_financiacion=financiacion,
                        coste_prestamo_titulos=obtener("coste_prestamo_titulos", 0.0) if cortos else 0.0,
                        permite_cortos=cortos)
