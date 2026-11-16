"""
Servicio de ejecución de bot de trading.
Migrado de tfm/app/services/trading/bot_executor.py

Adaptado para usar StorageInterface en lugar de SQLAlchemy Session directa.
Compatible con cualquier backend de persistencia (TFM PostgreSQL, etc.)
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ypyquant.execution.broker_interface import BrokerInterface, OrderSide, OrderStatus, OrderType
from ypyquant.execution.paper_broker import PaperBroker
from ypyquant.execution.wallet_manager import WalletManager
from ypyquant.execution.trade_analytics import TradeAnalytics
from ypyquant.execution.storage_interface import StorageInterface
from ypyquant.portfolio.rebalance_service import (
    RebalanceService,
    Position,
    RebalanceOperation,
    OperationType,
)

logger = logging.getLogger(__name__)


class BotExecutor:
    """
    Executor de bot de trading.

    Responsabilidades:
    1. Cargar portafolio via StorageInterface
    2. Conectar con el broker
    3. Orquestar el rebalanceo usando RebalanceService
    4. Ejecutar operaciones en el broker
    5. Registrar operaciones via StorageInterface
    """

    def __init__(
        self,
        storage: StorageInterface,
        portafolio_id: int,
        broker: Optional[BrokerInterface] = None,
        broker_config: Optional[Dict] = None,
        rebalance_threshold: float = 0.05,
    ):
        """
        Args:
            storage: Implementación de StorageInterface (provista por TFM)
            portafolio_id: ID del portafolio a ejecutar
            broker: Instancia de broker (usa PaperBroker si no se provee)
            broker_config: Configuración del broker
            rebalance_threshold: Umbral de rebalanceo (default: 5%)
        """
        self.storage = storage
        self.portafolio_id = portafolio_id

        # Cargar portafolio
        self.portafolio = self._load_portafolio()

        # Inicializar broker
        if broker is not None:
            self.broker = broker
        else:
            default_config = broker_config or {
                "initial_cash": self.portafolio.get("capital_actual") or 10000.0,
                "currency": "USD",
                "commission": 0.001,
            }
            self.broker = PaperBroker(default_config)
            logger.info("Usando PaperBroker (simulación)")

        # Inicializar servicio de rebalanceo
        self.rebalance_service = RebalanceService(
            rebalance_threshold=self.portafolio.get("rebalance_threshold") or rebalance_threshold,
            min_trade_value=100.0,
            commission_rate=0.001,
        )

        # Inicializar wallet manager
        self.wallet_manager = WalletManager(
            initial_capital=self.portafolio.get("capital_actual") or 10000.0,
            tradable_balance_ratio=0.99,
            max_open_trades=10,
            reserve_amount=0.0,
            max_position_size=0.30,
        )

        # Inicializar analytics
        self.trade_analytics = TradeAnalytics(storage=storage)

        # Conectar al broker
        self.broker.connect()

        logger.info(
            f"BotExecutor inicializado para portafolio "
            f"'{self.portafolio.get('nombre')}' (ID: {self.portafolio_id})"
        )

    def _load_portafolio(self) -> Dict:
        """Carga el portafolio via StorageInterface."""
        portafolio = self.storage.load_portafolio(self.portafolio_id)
        if portafolio.get("type") != "real":
            logger.warning(
                f"Portafolio '{portafolio.get('nombre')}' es tipo '{portafolio.get('type')}'. "
                f"Considera usar type='real' para trading en vivo."
            )
        return portafolio

    def execute_rebalance(
        self,
        target_weights: Optional[Dict[str, float]] = None,
        dry_run: bool = False,
    ) -> Dict:
        """
        Ejecuta un rebalanceo del portafolio.

        Args:
            target_weights: Pesos objetivo (usa los del portafolio si None)
            dry_run: Si es True, calcula sin ejecutar

        Returns:
            Dict con resultados del rebalanceo
        """
        logger.info("=" * 70)
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Ejecutando rebalanceo de '{self.portafolio.get('nombre')}'")
        logger.info("=" * 70)

        if target_weights is None:
            target_weights = self.portafolio.get("target_weights")

        if not target_weights:
            return {
                "success": False,
                "error": "No hay target_weights definidos en el portafolio",
                "operations": [],
            }

        account_info = self.broker.get_account_info()
        broker_positions = self.broker.get_positions()

        logger.info(f"Cash disponible: ${account_info['cash']:,.2f}")
        logger.info(f"Equity total: ${account_info['equity']:,.2f}")

        current_positions = [
            Position(ticker=pos["ticker"], quantity=pos["quantity"], price=pos["current_price"])
            for pos in broker_positions
        ]

        current_prices = {pos.ticker: pos.price for pos in current_positions}
        for ticker in target_weights:
            if ticker not in current_prices:
                price = self.broker.get_current_price(ticker)
                if price:
                    current_prices[ticker] = price

        rebalance_result = self.rebalance_service.calculate_rebalance(
            current_positions=current_positions,
            target_weights=target_weights,
            available_cash=account_info["cash"],
            current_prices=current_prices,
        )

        operations = rebalance_result["operations"]
        summary = rebalance_result["summary"]

        logger.info(
            f"Operaciones planificadas: {len(operations)} "
            f"({summary['buy_operations']} compras, {summary['sell_operations']} ventas)"
        )

        feasibility = self.rebalance_service.validate_operations_feasibility(
            operations=operations,
            available_cash=account_info["cash"],
            current_positions=current_positions,
        )

        if not feasibility["feasible"]:
            for error in feasibility["errors"]:
                logger.error(f"  - {error}")
            return {
                "success": False,
                "error": "Operations not feasible",
                "feasibility": feasibility,
                "rebalance_result": rebalance_result,
            }

        for warning in feasibility.get("warnings", []):
            logger.warning(f"  - {warning}")

        if dry_run:
            logger.info("[DRY RUN] No se ejecutarán operaciones")
            return {
                "success": True,
                "dry_run": True,
                "rebalance_result": rebalance_result,
                "feasibility": feasibility,
            }

        execution_result = self._execute_operations(operations)
        self._sync_positions_to_db()

        new_equity = self.broker.get_account_info()["equity"]
        self.storage.update_portafolio(self.portafolio_id, new_equity, datetime.utcnow())

        logger.info("=" * 70)
        logger.info("Rebalanceo completado")
        logger.info("=" * 70)

        return {
            "success": execution_result["success"],
            "rebalance_result": rebalance_result,
            "execution_result": execution_result,
            "feasibility": feasibility,
        }

    def _execute_operations(self, operations: List[RebalanceOperation]) -> Dict:
        executed_operations = []
        errors = []

        for op in operations:
            if op.operation_type == OperationType.BUY:
                order_side = OrderSide.BUY
            elif op.operation_type == OperationType.SELL:
                order_side = OrderSide.SELL
            else:
                continue  # HOLD

            logger.info(
                f"Ejecutando: {order_side.value.upper()} {op.quantity:.4f} {op.ticker} "
                f"@ ${op.current_price:.2f}"
            )

            try:
                order_result = self.broker.place_order(
                    ticker=op.ticker,
                    quantity=op.quantity,
                    side=order_side,
                    order_type=OrderType.MARKET,
                )

                if order_result["status"] == OrderStatus.FILLED:
                    op_id = self.storage.save_operation(self.portafolio_id, order_result)
                    executed_operations.append({
                        "operation_db_id": op_id,
                        "order_info": order_result,
                        "rebalance_operation": op,
                    })
                    logger.info(f"  ✓ Ejecutada y guardada en BD (ID: {op_id})")
                else:
                    error_msg = f"Orden rechazada para {op.ticker}: {order_result.get('error', 'Unknown')}"
                    errors.append(error_msg)
                    logger.error(f"  ✗ {error_msg}")

            except Exception as e:
                error_msg = f"Error ejecutando {order_side.value} {op.ticker}: {e}"
                errors.append(error_msg)
                logger.error(f"  ✗ {error_msg}")

        return {
            "success": len(errors) == 0,
            "operations_executed": executed_operations,
            "errors": errors,
            "summary": {
                "total_planned": len(operations),
                "total_executed": len(executed_operations),
                "total_failed": len(errors),
            },
        }

    def _sync_positions_to_db(self):
        broker_positions = self.broker.get_positions()
        self.storage.sync_positions(self.portafolio_id, broker_positions)
        logger.info(f"Sincronizadas {len(broker_positions)} posiciones en BD")

    def get_portfolio_status(self) -> Dict:
        account_info = self.broker.get_account_info()
        positions = self.broker.get_positions()

        current_weights = {}
        if account_info["equity"] > 0:
            for pos in positions:
                current_weights[pos["ticker"]] = pos["market_value"] / account_info["equity"]

        return {
            "portafolio_id": self.portafolio_id,
            "portafolio_nombre": self.portafolio.get("nombre"),
            "portafolio_tipo": self.portafolio.get("type"),
            "account_info": account_info,
            "positions": positions,
            "target_weights": self.portafolio.get("target_weights"),
            "current_weights": current_weights,
            "needs_rebalance": self._check_if_needs_rebalance(current_weights),
        }

    def _check_if_needs_rebalance(self, current_weights: Dict[str, float]) -> bool:
        target_weights = self.portafolio.get("target_weights") or {}
        threshold = self.portafolio.get("rebalance_threshold") or 0.05
        all_tickers = set(target_weights) | set(current_weights)
        return any(
            abs(target_weights.get(t, 0.0) - current_weights.get(t, 0.0)) > threshold
            for t in all_tickers
        )

    def get_operation_history(self, limit: int = 50) -> List[Dict]:
        return self.storage.get_operations(self.portafolio_id, limit=limit)

    def get_portfolio_metrics(self) -> Dict:
        positions = self.broker.get_positions()
        current_prices = {pos["ticker"]: pos["current_price"] for pos in positions}
        metrics = self.trade_analytics.calculate_portfolio_metrics(
            portafolio_id=self.portafolio_id,
            current_prices=current_prices,
        )
        return {
            "total_trades": metrics.total_trades,
            "open_trades": metrics.open_trades,
            "closed_trades": metrics.closed_trades,
            "total_pnl": metrics.total_pnl,
            "realized_pnl": metrics.total_realized_pnl,
            "unrealized_pnl": metrics.total_unrealized_pnl,
            "win_rate": metrics.win_rate,
            "avg_win": metrics.avg_win,
            "avg_loss": metrics.avg_loss,
            "profit_factor": metrics.profit_factor,
            "total_fees": metrics.total_fees,
            "best_trade": metrics.best_trade_pnl,
            "worst_trade": metrics.worst_trade_pnl,
            "total_return_pct": metrics.total_return_pct,
            "avg_return_pct": metrics.avg_return_pct,
            "summary": self.trade_analytics.format_portfolio_summary(metrics),
        }

    def get_wallet_status(self) -> Dict:
        account_info = self.broker.get_account_info()
        positions = self.broker.get_positions()
        used_capital = sum(pos["market_value"] for pos in positions)

        self.wallet_manager.update_balance(
            total_capital=account_info["equity"],
            used_capital=used_capital,
        )
        allocation = self.wallet_manager.get_portfolio_allocation()

        return {
            "balance": {
                "total": allocation["total_capital"],
                "free": allocation["free_capital"],
                "used": allocation["used_capital"],
                "tradable": allocation["tradable_balance"],
                "reserved": allocation["reserved_capital"],
            },
            "allocation": {
                "allocation_ratio": allocation["allocation_ratio"],
                "free_ratio": allocation["free_ratio"],
                "max_position_value": allocation["max_position_value"],
            },
            "limits": {
                "max_position_size_pct": self.wallet_manager.max_position_size,
                "max_open_trades": self.wallet_manager.max_open_trades,
                "tradable_balance_ratio": self.wallet_manager.tradable_balance_ratio,
            },
            "capacity": {
                "estimated_max_trades": self.wallet_manager.estimate_max_trades(),
                "can_open_more_trades": len(positions) < self.wallet_manager.max_open_trades,
            },
        }

    def validate_operation(
        self,
        ticker: str,
        stake_amount: float,
        min_stake: float = 10.0,
    ) -> Tuple[bool, str]:
        account_info = self.broker.get_account_info()
        positions = self.broker.get_positions()
        used_capital = sum(pos["market_value"] for pos in positions)
        self.wallet_manager.update_balance(
            total_capital=account_info["equity"],
            used_capital=used_capital,
        )
        return self.wallet_manager.validate_stake_amount(
            stake_amount=stake_amount,
            min_stake=min_stake,
            current_open_trades=len(positions),
        )

    def get_position_analytics(self, ticker: str) -> Optional[Dict]:
        posicion = self.storage.get_posicion(self.portafolio_id, ticker)
        if not posicion:
            return None

        current_price = self.broker.get_current_price(ticker)
        if not current_price:
            logger.warning(f"No se pudo obtener precio actual para {ticker}")
            return None

        metrics = self.trade_analytics.get_position_metrics(
            posicion=posicion,
            current_price=current_price,
        )
        return {
            "ticker": metrics.ticker,
            "entry_price": metrics.entry_price,
            "current_price": current_price,
            "quantity": metrics.quantity,
            "cost_basis": metrics.cost_basis,
            "current_value": metrics.current_value,
            "unrealized_pnl": metrics.unrealized_pnl,
            "unrealized_pnl_pct": metrics.return_pct,
            "total_fees": metrics.total_fees,
            "holding_days": metrics.holding_period_days,
            "summary": self.trade_analytics.format_trade_summary(metrics),
        }

    def close(self):
        if self.broker:
            self.broker.disconnect()
        logger.info("BotExecutor cerrado")
