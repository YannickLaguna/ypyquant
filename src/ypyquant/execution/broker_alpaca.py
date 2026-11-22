"""
broker_alpaca.py — Adaptador del puerto `Broker` para la cuenta paper de Alpaca (alpaca-py).

Responsabilidad:
    Traduce el puerto (broker.py) a llamadas de `TradingClient` (cuenta, posiciones, órdenes, reloj; también la
    capacidad de margen de la cuenta: `multiplier` y `shorting_enabled`) y,
    opcionalmente, de `StockHistoricalDataClient` (última cotización). Toda la API de alpaca-py
    (enums, peticiones) queda aquí: el resto del sistema no la importa. alpaca-py (extra `trading`) se
    importa en el momento de usarse, así que importar este módulo no exige el extra.
    Los errores de red se propagan, salvo `mercado_abierto` (ante error devuelve False: no se opera).

Se relaciona con:
    - ypyquant.execution.broker: el puerto que implementa; ejecutor.py: `EstadoCuenta`.
    - pipelines/paper_trading/alpaca_vivo.py: crea el adaptador con los clientes de `crear_clientes`.
    - app/api/paper.py: `ServicioAlpaca` sigue leyendo (cuenta, órdenes ALL, historial de equity) para la
      UI con su propio formato; no pasa por el puerto porque no es ejecución.
"""

import logging
from datetime import datetime

from ypyquant.execution.broker import EstadoOrden
from ypyquant.execution.ejecutor import COMPRA, EstadoCuenta
from ypyquant.execution.margen import SIN_MARGEN, CapacidadMargen

logger = logging.getLogger(__name__)


def _valor_enum(valor) -> str:
    """`OrderStatus.FILLED` o 'filled' -> 'filled'."""
    return str(getattr(valor, "value", valor)).lower()


class BrokerAlpaca:
    """
    Broker sobre Alpaca.

    Args:
        cliente: `TradingClient` (o un doble con los mismos métodos: get_clock, get_account,
            get_all_positions, get_orders, submit_order, get_order_by_id).
        cliente_datos: `StockHistoricalDataClient` opcional, solo para `precio_actual`.
    """

    def __init__(self, cliente, cliente_datos=None):
        self.cliente = cliente
        self.cliente_datos = cliente_datos

    def mercado_abierto(self) -> bool:
        """True si Alpaca informa de mercado abierto; ante error, False (no se opera)."""
        try:
            return bool(self.cliente.get_clock().is_open)
        except Exception as e:  # noqa: BLE001 - sin reloj no se opera
            logger.warning("No se pudo comprobar el estado del mercado: %s", e)
            return False

    def estado_cuenta(self) -> EstadoCuenta:
        cuenta = self.cliente.get_account()
        posiciones = {p.symbol: float(p.qty) for p in self.cliente.get_all_positions()}
        return EstadoCuenta(equity=float(cuenta.equity), efectivo=float(cuenta.cash), posiciones=posiciones)

    def capacidad_margen(self) -> CapacidadMargen:
        """
        Lo que la cuenta admite según Alpaca: `multiplier` ('1' sin margen, '2' margen, '4' intradía) y `shorting_enabled`
        del objeto cuenta de alpaca-py. Una cuenta sin `multiplier` informado se trata como sin margen.
        """
        cuenta = self.cliente.get_account()
        try:
            multiplicador = float(getattr(cuenta, "multiplier", None) or 1.0)
        except (TypeError, ValueError):
            return SIN_MARGEN
        return CapacidadMargen(margen=multiplicador > 1.0, multiplicador=multiplicador,
                               cortos=bool(getattr(cuenta, "shorting_enabled", False)))

    def tickers_con_orden_abierta(self) -> set[str]:
        """Tickers con alguna orden aún abierta (nueva, aceptada, parcial...). Propaga el error de red."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        abiertas = self.cliente.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        return {orden.symbol for orden in abiertas}

    def enviar_orden(self, ticker: str, lado: str, cantidad: float) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        orden = self.cliente.submit_order(MarketOrderRequest(
            symbol=ticker, qty=cantidad, side=OrderSide.BUY if lado == COMPRA else OrderSide.SELL,
            time_in_force=TimeInForce.DAY))
        return str(orden.id)

    def consultar_orden(self, id_orden: str) -> EstadoOrden:
        orden = self.cliente.get_order_by_id(id_orden)
        fecha = orden.filled_at if isinstance(orden.filled_at, datetime) else None
        return EstadoOrden(
            id=str(orden.id), ticker=orden.symbol,
            lado=COMPRA if _valor_enum(orden.side) == "buy" else "venta",
            estado=_valor_enum(orden.status), cantidad_ejecutada=float(orden.filled_qty or 0),
            precio_medio=float(orden.filled_avg_price or 0), fecha_ejecucion=fecha,
        )

    def precio_actual(self, ticker: str) -> float | None:
        """Última cotización (ask, si no bid) del cliente de datos; None si no hay cliente o precio."""
        if self.cliente_datos is None:
            return None
        from alpaca.data.requests import StockLatestQuoteRequest

        cotizacion = self.cliente_datos.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))
        precio = float(cotizacion[ticker].ask_price or cotizacion[ticker].bid_price or 0)
        return precio if precio > 0 else None
