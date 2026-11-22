"""
broker.py — Puerto (seam) de ejecución: lo que el sistema necesita de un broker, sin nombrar a ninguno.

Responsabilidad:
    Define el `Broker` (Protocol) y los datos que cruzan la costura: `EstadoOrden`. Es la interfaz
    pequeña tras la que viven los detalles de cada broker (clientes, enums, reintentos, formatos):

        mercado_abierto() -> bool
        estado_cuenta() -> EstadoCuenta                 (equity, efectivo, posiciones)
        tickers_con_orden_abierta() -> set[str]         (propaga el error de red: quien llama decide)
        enviar_orden(ticker, lado, cantidad) -> id      (orden de mercado válida solo ese día)
        consultar_orden(id) -> EstadoOrden              (estado y fill real)
        precio_actual(ticker) -> float | None           (cotización de referencia)

    Adaptadores: `BrokerSimulado` (en memoria: paper interno y dobles de test) y `BrokerAlpaca`
    (cuenta paper de Alpaca). Interactive Brokers queda como adaptador futuro: necesitará traducir
    ticker -> contrato (conid/exchange/divisa), esperar la conexión con TWS/Gateway (asíncrona),
    mapear los estados de orden de IB a `ESTADOS_TERMINALES` y gestionar el horario del mercado por
    contrato; el puerto no cambia.

    Estados de orden en minúscula, como Alpaca: new, accepted, partially_filled, filled, canceled,
    expired, rejected, done_for_day, replaced. Los de `ESTADOS_TERMINALES` ya no cambian.

Se relaciona con:
    - ypyquant.execution.ejecutor: aporta `EstadoCuenta` y las órdenes que se envían por este puerto.
    - ypyquant.execution.broker_simulado y broker_alpaca: los adaptadores reales.
    - pipelines/paper_trading/alpaca_vivo.py: orquestación (calendario -> estrategia -> Ejecutor -> broker).
    - ypyquant.execution.nautilus.registro_fills: consume `EstadoOrden.a_dict_fill()`.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from ypyquant.execution.ejecutor import COMPRA, EstadoCuenta

# Estados tras los cuales una orden ya no cambia (si hay cantidad ejecutada, ese fill es el definitivo)
ESTADOS_TERMINALES = frozenset({"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"})
# Estados en los que la orden sigue viva en el broker (bloquean nuevas órdenes sobre ese ticker)
ESTADOS_ABIERTOS = frozenset({"new", "accepted", "partially_filled", "pending_new"})


@dataclass(frozen=True)
class EstadoOrden:
    """Estado de una orden enviada: `estado` (ver cabecera), cantidad ejecutada y precio medio de ejecución."""
    id: str
    ticker: str
    lado: str  # "compra" | "venta"
    estado: str
    cantidad_ejecutada: float = 0.0
    precio_medio: float = 0.0
    fecha_ejecucion: datetime | None = None

    @property
    def terminal(self) -> bool:
        return self.estado in ESTADOS_TERMINALES

    def a_dict_fill(self) -> dict:
        """Formato que entiende `RegistroFills.registrar` (el de un order de Alpaca)."""
        return {
            "id": self.id, "symbol": self.ticker, "side": "buy" if self.lado == COMPRA else "sell",
            "filled_qty": str(self.cantidad_ejecutada), "filled_avg_price": str(self.precio_medio),
            "filled_at": self.fecha_ejecucion.isoformat() if self.fecha_ejecucion else None,
        }


@runtime_checkable
class Broker(Protocol):
    """Puerto de ejecución: ver la cabecera del módulo."""

    def mercado_abierto(self) -> bool: ...

    def estado_cuenta(self) -> EstadoCuenta: ...

    def tickers_con_orden_abierta(self) -> set[str]: ...

    def enviar_orden(self, ticker: str, lado: str, cantidad: float) -> str: ...

    def consultar_orden(self, id_orden: str) -> EstadoOrden: ...

    def precio_actual(self, ticker: str) -> float | None: ...
