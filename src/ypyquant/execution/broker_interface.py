"""
Interfaz abstracta para brokers.
Define los métodos que cualquier implementación de broker debe cumplir.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    """Tipos de órdenes soportadas"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Lado de la orden"""
    BUY = "compra"
    SELL = "venta"


class OrderStatus(Enum):
    """Estados de una orden"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class BrokerInterface(ABC):
    """
    Interfaz abstracta para conectarse con diferentes brokers.

    Implementaciones concretas: PaperBroker (simulación), AlpacaBroker, IBBroker.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.connected = False

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> bool: ...

    @abstractmethod
    def get_account_info(self) -> Dict:
        """
        Returns:
            {'account_id', 'cash', 'equity', 'buying_power', 'currency'}
        """
        ...

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """
        Returns:
            [{'ticker', 'quantity', 'avg_entry_price', 'current_price', 'market_value', 'unrealized_pnl'}, ...]
        """
        ...

    @abstractmethod
    def place_order(
        self,
        ticker: str,
        quantity: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Dict:
        """
        Returns:
            {'order_id', 'status', 'ticker', 'quantity', 'side', 'order_type',
             'filled_quantity', 'filled_price', 'created_at', 'filled_at'}
        """
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_current_price(self, ticker: str) -> Optional[float]: ...

    def is_market_open(self) -> bool:
        """Verifica si el mercado NYSE está abierto (9:30–16:00 ET, L–V)."""
        import pytz

        eastern = pytz.timezone("US/Eastern")
        now = datetime.now(eastern)

        if now.weekday() >= 5:
            return False

        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close
