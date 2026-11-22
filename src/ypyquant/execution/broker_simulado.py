"""
broker_simulado.py — Adaptador del puerto `Broker` en memoria (paper interno y dobles de test).

Responsabilidad:
    Cuenta simulada: efectivo, posiciones largas y órdenes de mercado que se ejecutan al precio de
    referencia con comisión y slippage (mismo modelo que el motor de backtest: comisión sobre el
    importe, slippage en el precio de ejecución). Es la base del paper interno (fase 3 del plan) y el
    fake con el que se prueba el orquestador sin red.
    - Precios y reloj del mercado los fija quien lo usa (`fijar_precios`, `fijar_mercado`).
    - `ejecucion_inmediata=True` (por defecto): la orden se llena al enviarla. Con False queda abierta
      (estado `new`, bloquea el ticker) hasta `ejecutar_pendientes()`: sirve para probar el bloqueo
      por órdenes abiertas y los fills diferidos.
    - Rechaza (estado `rejected`) compras sin efectivo suficiente, ventas por encima de la posición y
      órdenes sobre tickers sin precio. `fallar_consulta_abiertas` simula una caída de red.
    - Margen y cortos (opcional, `reglas_margen` de `ypyquant.execution.margen`): con reglas activas una compra puede
      superar el efectivo hasta el límite de exposición bruta (el efectivo queda negativo = préstamo), una venta sin
      posición larga abre un corto (el producto entra en efectivo) y una compra sobre un corto lo cubre. Las mismas
      reglas que el motor; sin reglas (por defecto) el broker es solo largo y sin apalancamiento, como siempre.
      Una orden no cruza por cero: vender más que la posición larga (o cubrir más que el corto) se rechaza.

Se relaciona con:
    - ypyquant.execution.broker: el puerto que implementa; ejecutor.py: `EstadoCuenta`.
    - tests/test_ejecutor_orquestador.py: orquestador del bot con este adaptador.
"""

import itertools
from collections.abc import Mapping
from datetime import UTC, datetime

from ypyquant.execution.broker import EstadoOrden
from ypyquant.execution.ejecutor import COMPRA, VENTA, EstadoCuenta
from ypyquant.execution.margen import ReglasMargen, capacidad_bruta, capacidad_de_compra, valores_largo_corto


class BrokerSimulado:
    """Broker en memoria; ver la cabecera del módulo."""

    def __init__(self, efectivo_inicial: float = 100_000.0, comision: float = 0.0, slippage: float = 0.0,
                 precios: Mapping[str, float] | None = None, mercado_abierto: bool = True,
                 ejecucion_inmediata: bool = True, reglas_margen: ReglasMargen | None = None):
        self.reglas = reglas_margen or ReglasMargen()
        # Equity con el que se mide la capacidad de margen (None = el actual de la cuenta). El simulador del paper lo fija
        # a la foto previa al rebalanceo, como el motor, para que ambos recorten igual las órdenes que rozan el límite.
        self.equity_referencia: float | None = None
        self.efectivo = float(efectivo_inicial)
        self.posiciones: dict[str, float] = {}
        self.comision = comision
        self.slippage = slippage
        self.precios: dict[str, float] = dict(precios or {})
        self.abierto = mercado_abierto
        self.ejecucion_inmediata = ejecucion_inmediata
        self.fallar_consulta_abiertas = False
        self.ordenes: dict[str, EstadoOrden] = {}
        self.enviadas: list[tuple[str, str, float]] = []  # (ticker, lado, cantidad) en orden de envío
        self._cantidades: dict[str, float] = {}  # cantidad pedida de cada orden
        self._ids = itertools.count(1)

    # --- control de la simulación ---

    def fijar_precios(self, precios: Mapping[str, float]) -> None:
        self.precios.update(precios)

    def fijar_mercado(self, abierto: bool) -> None:
        self.abierto = abierto

    def ejecutar_pendientes(self) -> None:
        """Llena las órdenes que quedaron abiertas (modo `ejecucion_inmediata=False`)."""
        for id_orden, orden in list(self.ordenes.items()):
            if orden.estado == "new":
                self._llenar(id_orden, orden, self._cantidades[id_orden])

    # --- puerto Broker ---

    def mercado_abierto(self) -> bool:
        return self.abierto

    def estado_cuenta(self) -> EstadoCuenta:
        valor = sum(q * self.precios.get(t, 0.0) for t, q in self.posiciones.items())
        return EstadoCuenta(equity=self.efectivo + valor, efectivo=self.efectivo, posiciones=dict(self.posiciones))

    def tickers_con_orden_abierta(self) -> set[str]:
        if self.fallar_consulta_abiertas:
            raise ConnectionError("sin red (simulado)")
        return {o.ticker for o in self.ordenes.values() if o.estado == "new"}

    def enviar_orden(self, ticker: str, lado: str, cantidad: float) -> str:
        id_orden = f"sim-{next(self._ids)}"
        self.enviadas.append((ticker, lado, cantidad))
        self._cantidades[id_orden] = cantidad
        orden = EstadoOrden(id_orden, ticker, lado, "new")
        self.ordenes[id_orden] = orden
        if self.ejecucion_inmediata:
            self._llenar(id_orden, orden, cantidad)
        return id_orden

    def consultar_orden(self, id_orden: str) -> EstadoOrden:
        return self.ordenes[str(id_orden)]

    def precio_actual(self, ticker: str) -> float | None:
        return self.precios.get(ticker)

    # --- ejecución ---

    def _llenar(self, id_orden: str, orden: EstadoOrden, cantidad: float) -> None:
        """Ejecuta al precio de referencia con slippage y comisión, o rechaza si no es viable."""
        precio = self.precios.get(orden.ticker)
        if precio is None or precio <= 0 or cantidad <= 0:
            self.ordenes[id_orden] = EstadoOrden(id_orden, orden.ticker, orden.lado, "rejected")
            return
        posicion = self.posiciones.get(orden.ticker, 0.0)
        if orden.lado == COMPRA:
            precio_ejecucion = precio * (1 + self.slippage)
            coste = cantidad * precio_ejecucion * (1 + self.comision)
            if posicion < 0:  # cubrir un corto (sin pasar de la posición: no se cruza por cero)
                if cantidad > -posicion + 1e-9:
                    self._rechazar(id_orden, orden)
                    return
            elif coste > self._capacidad_de_compra() + 1e-9:
                self._rechazar(id_orden, orden)
                return
            self.efectivo -= coste
            nueva = posicion + cantidad
            if abs(nueva) > 1e-9:
                self.posiciones[orden.ticker] = nueva
            else:
                self.posiciones.pop(orden.ticker, None)
        else:
            precio_ejecucion = precio * (1 - self.slippage)
            if posicion > 0:  # vender largo (sin pasar de la posición)
                if cantidad > posicion + 1e-9:
                    self._rechazar(id_orden, orden)
                    return
            elif not self.reglas.permite_cortos or cantidad * precio_ejecucion > self._capacidad_bruta() + 1e-9:
                self._rechazar(id_orden, orden)  # sin posición larga solo se vende en corto, con margen y dentro del límite
                return
            self.efectivo += cantidad * precio_ejecucion * (1 - self.comision)
            nueva = posicion - cantidad
            if abs(nueva) > 1e-9:
                self.posiciones[orden.ticker] = nueva
            else:
                self.posiciones.pop(orden.ticker, None)
        self.ordenes[id_orden] = EstadoOrden(id_orden, orden.ticker, orden.lado, "filled", cantidad,
                                             precio_ejecucion, datetime.now(UTC))

    # --- margen ---

    def _valores(self) -> tuple[float, float, float]:
        """`(equity, valor de largos, valor de cortos)` con los precios actuales."""
        largo, corto = valores_largo_corto(self.posiciones, self.precios)
        equity = self.equity_referencia if self.equity_referencia is not None else self.efectivo + largo - corto
        return equity, largo, corto

    def _capacidad_de_compra(self) -> float:
        """Importe que se puede gastar en una compra (el efectivo si no hay reglas de margen activas)."""
        equity, largo, corto = self._valores()
        return capacidad_de_compra(self.efectivo, equity, largo, corto, self.reglas)

    def _capacidad_bruta(self) -> float:
        equity, largo, corto = self._valores()
        return capacidad_bruta(equity, largo, corto, self.reglas)

    def _rechazar(self, id_orden: str, orden: EstadoOrden) -> None:
        self.ordenes[id_orden] = EstadoOrden(id_orden, orden.ticker, orden.lado, "rejected")
