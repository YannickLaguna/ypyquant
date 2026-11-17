"""
Implementación de broker de papel (Paper Trading).
Simula operaciones sin conexión a un broker real.
Migrado de tfm/app/services/trading/paper_broker.py
"""

import logging
import uuid
from datetime import UTC, datetime

from ypyquant.execution.broker_interface import (
    BrokerInterface,
    OrderSide,
    OrderStatus,
    OrderType,
)

logger = logging.getLogger(__name__)


class PaperBroker(BrokerInterface):
    """
    Broker de papel que simula operaciones sin ejecutarlas realmente.
    Mantiene un estado virtual de cuenta, posiciones y órdenes.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: {
                'initial_cash': float,   # Capital inicial (default: 10000)
                'currency': str,         # Moneda (default: 'USD')
                'commission': float      # Tasa de comisión (default: 0.001 = 0.1%)
            }
        """
        super().__init__(config)
        self.cash = config.get("initial_cash", 10000.0)
        self.initial_cash = self.cash
        self.currency = config.get("currency", "USD")
        self.commission_rate = config.get("commission", 0.001)

        self.positions: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self.order_history: list[dict] = []

        logger.info(f"PaperBroker inicializado con ${self.cash:.2f} {self.currency}")

    def connect(self) -> bool:
        self.connected = True
        logger.info("PaperBroker: Conexión establecida (simulada)")
        return True

    def disconnect(self) -> bool:
        self.connected = False
        logger.info("PaperBroker: Desconectado")
        return True

    def get_account_info(self) -> dict:
        equity = self.cash
        for ticker, position in self.positions.items():
            current_price = self.get_current_price(ticker)
            if current_price:
                equity += position["quantity"] * current_price

        return {
            "account_id": "PAPER_" + str(uuid.uuid4())[:8],
            "cash": self.cash,
            "equity": equity,
            "buying_power": self.cash,
            "currency": self.currency,
            "positions_count": len(self.positions),
            "initial_cash": self.initial_cash,
            "total_return": ((equity - self.initial_cash) / self.initial_cash) * 100,
        }

    def get_positions(self) -> list[dict]:
        positions_list = []
        for ticker, position in self.positions.items():
            current_price = self.get_current_price(ticker) or position["avg_price"]
            market_value = position["quantity"] * current_price
            cost_basis = position["quantity"] * position["avg_price"]
            unrealized_pnl = market_value - cost_basis

            positions_list.append({
                "ticker": ticker,
                "quantity": position["quantity"],
                "avg_entry_price": position["avg_price"],
                "current_price": current_price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0,
            })
        return positions_list

    def place_order(
        self,
        ticker: str,
        quantity: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> dict:
        if not self.connected:
            raise RuntimeError("PaperBroker no está conectado")

        order_id = f"PAPER_{uuid.uuid4().hex[:12].upper()}"
        current_price = self.get_current_price(ticker)

        if current_price is None:
            logger.error(f"No se pudo obtener precio para {ticker}")
            return {
                "order_id": order_id,
                "status": OrderStatus.REJECTED,
                "ticker": ticker,
                "quantity": quantity,
                "side": side.value,
                "order_type": order_type.value,
                "error": "Could not fetch current price",
                "created_at": datetime.now(UTC),
            }

        if order_type != OrderType.MARKET:
            order_type = OrderType.MARKET

        filled_price = current_price
        commission = quantity * filled_price * self.commission_rate

        try:
            if side == OrderSide.BUY:
                self._execute_buy(ticker, quantity, filled_price, commission)
            else:
                self._execute_sell(ticker, quantity, filled_price, commission)

            order_info = {
                "order_id": order_id,
                "status": OrderStatus.FILLED,
                "ticker": ticker,
                "quantity": quantity,
                "side": side.value,
                "order_type": order_type.value,
                "filled_quantity": quantity,
                "filled_price": filled_price,
                "commission": commission,
                "created_at": datetime.now(UTC),
                "filled_at": datetime.now(UTC),
            }

            self.orders[order_id] = order_info
            self.order_history.append(order_info)
            logger.info(
                f"PaperBroker: {side.value} {quantity} {ticker} @ ${filled_price:.2f} "
                f"(comisión: ${commission:.2f})"
            )
            return order_info

        except Exception as e:
            logger.error(f"Error ejecutando orden: {e}")
            return {
                "order_id": order_id,
                "status": OrderStatus.REJECTED,
                "ticker": ticker,
                "quantity": quantity,
                "side": side.value,
                "order_type": order_type.value,
                "error": str(e),
                "created_at": datetime.now(UTC),
            }

    def _execute_buy(self, ticker: str, quantity: float, price: float, commission: float):
        total_cost = quantity * price + commission
        if total_cost > self.cash:
            raise ValueError(
                f"Fondos insuficientes: ${self.cash:.2f} disponibles, ${total_cost:.2f} requeridos"
            )
        self.cash -= total_cost

        if ticker in self.positions:
            pos = self.positions[ticker]
            total_qty = pos["quantity"] + quantity
            total_basis = pos["quantity"] * pos["avg_price"] + quantity * price
            self.positions[ticker] = {"quantity": total_qty, "avg_price": total_basis / total_qty}
        else:
            self.positions[ticker] = {"quantity": quantity, "avg_price": price}

    def _execute_sell(self, ticker: str, quantity: float, price: float, commission: float):
        if ticker not in self.positions:
            raise ValueError(f"No se tiene posición en {ticker}")

        current_qty = self.positions[ticker]["quantity"]
        if quantity > current_qty:
            raise ValueError(
                f"Cantidad insuficiente de {ticker}: {current_qty} disponibles, {quantity} solicitadas"
            )

        self.cash += quantity * price - commission
        remaining = current_qty - quantity

        if remaining > 1e-4:
            self.positions[ticker]["quantity"] = remaining
        else:
            del self.positions[ticker]
            logger.info(f"Posición en {ticker} cerrada completamente")

    def get_order_status(self, order_id: str) -> dict:
        return self.orders.get(
            order_id,
            {"order_id": order_id, "status": OrderStatus.REJECTED, "error": "Order not found"},
        )

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders and self.orders[order_id]["status"] == OrderStatus.FILLED:
            logger.warning(f"Orden {order_id} ya fue ejecutada, no se puede cancelar")
            return False
        return True

    def get_current_price(self, ticker: str) -> float | None:
        try:
            import yfinance as yf
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error obteniendo precio de {ticker}: {e}")
            return None

    def get_order_history(self) -> list[dict]:
        return self.order_history.copy()

    def reset(self):
        self.cash = self.initial_cash
        self.positions.clear()
        self.orders.clear()
        self.order_history.clear()
        logger.info("PaperBroker: Estado reseteado a condiciones iniciales")
