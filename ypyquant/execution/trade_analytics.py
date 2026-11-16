"""
Servicio de análisis de trades y cálculos de P&L.
Migrado de tfm/app/services/trading/trade_analytics.py

Adaptado para usar StorageInterface en lugar de SQLAlchemy Session directa.
Los métodos @staticmethod son independientes de almacenamiento y se pueden usar sin instanciar.
"""

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timezone
from dataclasses import dataclass

if TYPE_CHECKING:
    from ypyquant.execution.storage_interface import StorageInterface

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Métricas de un trade individual"""
    ticker: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float

    cost_basis: float
    current_value: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    return_pct: float

    entry_fee: float
    exit_fee: float
    total_fees: float

    entry_date: datetime
    exit_date: Optional[datetime]
    holding_period_days: Optional[float]

    is_open: bool


@dataclass
class PortfolioMetrics:
    """Métricas agregadas del portafolio"""
    total_trades: int
    open_trades: int
    closed_trades: int

    total_realized_pnl: float
    total_unrealized_pnl: float
    total_pnl: float

    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    total_fees: float
    best_trade_pnl: float
    worst_trade_pnl: float
    total_return_pct: float
    avg_return_pct: float


class TradeAnalytics:
    """
    Servicio de análisis de trades y cálculos de P&L.

    Acepta una StorageInterface para obtener datos de BD sin acoplarse al ORM concreto.
    Los métodos @staticmethod pueden usarse directamente sin instanciar la clase.
    """

    def __init__(self, storage: "StorageInterface"):
        self.storage = storage

    @staticmethod
    def calculate_trade_pnl(
        entry_price: float,
        exit_price: float,
        quantity: float,
        entry_fee: float = 0.0,
        exit_fee: float = 0.0,
        is_short: bool = False,
    ) -> Tuple[float, float]:
        """
        Calcula el P&L de un trade.

        Returns:
            (pnl_absolute, pnl_pct)
        """
        price_diff = (entry_price - exit_price) if is_short else (exit_price - entry_price)
        gross_pnl = price_diff * quantity
        net_pnl = gross_pnl - entry_fee - exit_fee
        cost_basis = entry_price * quantity + entry_fee
        pnl_pct = (net_pnl / cost_basis) if cost_basis > 0 else 0
        return net_pnl, pnl_pct

    @staticmethod
    def calculate_position_metrics(
        ticker: str,
        entry_price: float,
        current_price: float,
        quantity: float,
        entry_fee: float,
        entry_date: datetime,
        exit_price: Optional[float] = None,
        exit_fee: Optional[float] = None,
        exit_date: Optional[datetime] = None,
        is_short: bool = False,
    ) -> TradeMetrics:
        """Calcula métricas completas de una posición."""
        cost_basis = entry_price * quantity + entry_fee
        current_value = current_price * quantity

        if is_short:
            unrealized_pnl = (entry_price - current_price) * quantity - entry_fee
        else:
            unrealized_pnl = (current_price - entry_price) * quantity - entry_fee

        realized_pnl = 0.0
        if exit_price is not None:
            realized_pnl, _ = TradeAnalytics.calculate_trade_pnl(
                entry_price, exit_price, quantity, entry_fee, exit_fee or 0, is_short
            )

        total_pnl = realized_pnl if exit_price else unrealized_pnl
        return_pct = (total_pnl / cost_basis) if cost_basis > 0 else 0
        total_fees = entry_fee + (exit_fee or 0)

        if exit_date:
            holding_period_days = (exit_date - entry_date).total_seconds() / 86400
        else:
            now = datetime.now(timezone.utc)
            ed = entry_date if entry_date.tzinfo else entry_date.replace(tzinfo=timezone.utc)
            holding_period_days = (now - ed).total_seconds() / 86400

        return TradeMetrics(
            ticker=ticker,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            cost_basis=cost_basis,
            current_value=current_value,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            return_pct=return_pct,
            entry_fee=entry_fee,
            exit_fee=exit_fee or 0,
            total_fees=total_fees,
            entry_date=entry_date,
            exit_date=exit_date,
            holding_period_days=holding_period_days,
            is_open=exit_price is None,
        )

    def get_position_metrics(self, posicion: Dict, current_price: float) -> TradeMetrics:
        """
        Calcula métricas de una posición.

        Args:
            posicion: Dict con portafolio_id, ticker, precio_entrada, cantidad, fecha_entrada
            current_price: Precio actual del activo
        """
        ops = self.storage.get_operations(posicion["portafolio_id"], ticker=posicion["ticker"])
        buy_ops = [op for op in ops if op["tipo"] == "compra"]
        entry_fee = buy_ops[0]["comision"] if buy_ops else 0.0

        return self.calculate_position_metrics(
            ticker=posicion["ticker"],
            entry_price=posicion["precio_entrada"],
            current_price=current_price,
            quantity=posicion["cantidad"],
            entry_fee=entry_fee,
            entry_date=posicion["fecha_entrada"],
        )

    def get_trade_metrics_from_operations(
        self,
        ticker: str,
        portafolio_id: int,
        current_price: Optional[float] = None,
    ) -> List[TradeMetrics]:
        """
        Calcula métricas de trades completados desde operaciones de la BD.
        """
        operations = self.storage.get_operations(portafolio_id, ticker=ticker)
        if not operations:
            return []

        # Ordenar por fecha ascendente
        operations = sorted(operations, key=lambda op: op["fecha"])

        trades = []
        position_qty = 0.0
        avg_entry_price = 0.0
        total_cost = 0.0
        entry_date = None
        entry_fees = 0.0

        for op in operations:
            if op["tipo"] == "compra":
                total_cost += op["precio"] * op["cantidad"] + op["comision"]
                position_qty += op["cantidad"]
                entry_fees += op["comision"]
                if entry_date is None:
                    entry_date = op["fecha"]
                if position_qty > 0:
                    avg_entry_price = (total_cost - entry_fees) / position_qty

            elif op["tipo"] == "venta" and position_qty > 0:
                sell_qty = min(op["cantidad"], position_qty)
                metrics = self.calculate_position_metrics(
                    ticker=ticker,
                    entry_price=avg_entry_price,
                    current_price=op["precio"],
                    quantity=sell_qty,
                    entry_fee=entry_fees * (sell_qty / position_qty),
                    entry_date=entry_date,
                    exit_price=op["precio"],
                    exit_fee=op["comision"],
                    exit_date=op["fecha"],
                )
                trades.append(metrics)

                position_qty -= sell_qty
                if position_qty <= 0:
                    total_cost = 0.0
                    entry_fees = 0.0
                    entry_date = None
                else:
                    ratio = sell_qty / (position_qty + sell_qty)
                    total_cost -= avg_entry_price * sell_qty + entry_fees * ratio
                    entry_fees -= entry_fees * ratio

        if position_qty > 0 and current_price:
            metrics = self.calculate_position_metrics(
                ticker=ticker,
                entry_price=avg_entry_price,
                current_price=current_price,
                quantity=position_qty,
                entry_fee=entry_fees,
                entry_date=entry_date,
            )
            trades.append(metrics)

        return trades

    def calculate_portfolio_metrics(
        self,
        portafolio_id: int,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> PortfolioMetrics:
        """Calcula métricas agregadas del portafolio."""
        operations = self.storage.get_operations(portafolio_id)
        if not operations:
            return self._empty_portfolio_metrics()

        tickers = {op["ticker"] for op in operations}
        all_trades = []
        for ticker in tickers:
            current_price = (current_prices or {}).get(ticker)
            trades = self.get_trade_metrics_from_operations(ticker, portafolio_id, current_price)
            all_trades.extend(trades)

        if not all_trades:
            return self._empty_portfolio_metrics()

        open_trades = [t for t in all_trades if t.is_open]
        closed_trades = [t for t in all_trades if not t.is_open]

        total_realized_pnl = sum(t.realized_pnl for t in all_trades)
        total_unrealized_pnl = sum(t.unrealized_pnl for t in all_trades)
        total_pnl = total_realized_pnl + total_unrealized_pnl
        total_fees = sum(t.total_fees for t in all_trades)

        winning = [t for t in closed_trades if t.total_pnl > 0]
        losing = [t for t in closed_trades if t.total_pnl <= 0]

        win_rate = len(winning) / len(closed_trades) if closed_trades else 0
        avg_win = sum(t.total_pnl for t in winning) / len(winning) if winning else 0
        avg_loss = sum(t.total_pnl for t in losing) / len(losing) if losing else 0

        total_wins = sum(t.total_pnl for t in winning)
        total_losses = abs(sum(t.total_pnl for t in losing))
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        all_pnls = [t.total_pnl for t in all_trades]
        total_cost_basis = sum(t.cost_basis for t in all_trades)

        return PortfolioMetrics(
            total_trades=len(all_trades),
            open_trades=len(open_trades),
            closed_trades=len(closed_trades),
            total_realized_pnl=total_realized_pnl,
            total_unrealized_pnl=total_unrealized_pnl,
            total_pnl=total_pnl,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_fees=total_fees,
            best_trade_pnl=max(all_pnls) if all_pnls else 0,
            worst_trade_pnl=min(all_pnls) if all_pnls else 0,
            total_return_pct=total_pnl / total_cost_basis if total_cost_basis > 0 else 0,
            avg_return_pct=sum(t.return_pct for t in all_trades) / len(all_trades) if all_trades else 0,
        )

    @staticmethod
    def _empty_portfolio_metrics() -> PortfolioMetrics:
        return PortfolioMetrics(
            total_trades=0, open_trades=0, closed_trades=0,
            total_realized_pnl=0, total_unrealized_pnl=0, total_pnl=0,
            win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
            total_fees=0, best_trade_pnl=0, worst_trade_pnl=0,
            total_return_pct=0, avg_return_pct=0,
        )

    @staticmethod
    def format_trade_summary(metrics: TradeMetrics) -> str:
        status = "ABIERTO" if metrics.is_open else "CERRADO"
        sign = "+" if metrics.total_pnl >= 0 else ""
        exit_p = metrics.exit_price or (metrics.current_value / metrics.quantity if metrics.quantity else 0)
        return (
            f"Trade {status}: {metrics.ticker}\n"
            f"  Entrada: ${metrics.entry_price:.2f} x {metrics.quantity:.2f}\n"
            f"  Salida:  ${exit_p:.2f}\n"
            f"  P&L:     {sign}${metrics.total_pnl:,.2f} ({sign}{metrics.return_pct:.2%})\n"
            f"  Fees:    ${metrics.total_fees:.2f}  Holding: {metrics.holding_period_days:.1f}d"
        )

    @staticmethod
    def format_portfolio_summary(metrics: PortfolioMetrics) -> str:
        sign = "+" if metrics.total_pnl >= 0 else ""
        return (
            f"Portafolio: {metrics.total_trades} trades "
            f"({metrics.open_trades} abiertos, {metrics.closed_trades} cerrados)\n"
            f"P&L: {sign}${metrics.total_pnl:,.2f} ({sign}{metrics.total_return_pct:.2%})\n"
            f"Win rate: {metrics.win_rate:.1%}  Profit factor: {metrics.profit_factor:.2f}\n"
            f"Best: ${metrics.best_trade_pnl:,.2f}  Worst: ${metrics.worst_trade_pnl:,.2f}"
        )
