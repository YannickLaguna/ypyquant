"""
Tests de smoke para ypyquant.execution (M0, Fase 3+4).

Responsabilidad:
    Verificar la migración de TFM a ypyquant.execution: interfaz de broker, PaperBroker,
    ProtectionManager, WalletManager, TradeAnalytics, BotExecutor y API pública. Usa un
    almacenamiento simulado (MockStorage); no requiere BD ni broker real.

Se relaciona con:
    - src/ypyquant/execution/ (código bajo prueba).
    - Es unittest: pytest lo recoge tal cual (pyproject.toml, testpaths=tests).
"""

import unittest
from datetime import datetime

# --- StorageInterface mock para tests ---

class MockStorage:
    """Implementación mínima de StorageInterface para tests."""

    def load_portafolio(self, portafolio_id: int) -> dict:
        return {
            "nombre": "Test Portfolio",
            "type": "real",
            "capital_actual": 10000.0,
            "target_weights": {"AAPL": 0.5, "MSFT": 0.5},
            "rebalance_threshold": 0.05,
        }

    def update_portafolio(self, portafolio_id, capital_actual, last_optimization_date):
        pass

    def save_operation(self, portafolio_id, order_info) -> int:
        return 1

    def sync_positions(self, portafolio_id, positions):
        pass

    def get_operations(self, portafolio_id, ticker=None, limit=0) -> list[dict]:
        ops = [
            {
                "id": 1,
                "ticker": "AAPL",
                "tipo": "compra",
                "cantidad": 10.0,
                "precio": 150.0,
                "comision": 1.5,
                "fecha": datetime(2024, 1, 2),
            },
            {
                "id": 2,
                "ticker": "AAPL",
                "tipo": "venta",
                "cantidad": 5.0,
                "precio": 160.0,
                "comision": 0.8,
                "fecha": datetime(2024, 2, 1),
            },
        ]
        if ticker:
            ops = [op for op in ops if op["ticker"] == ticker]
        return ops[:limit] if limit else ops

    def get_posicion(self, portafolio_id, ticker) -> dict | None:
        if ticker == "AAPL":
            return {
                "portafolio_id": portafolio_id,
                "ticker": "AAPL",
                "precio_entrada": 150.0,
                "cantidad": 5.0,
                "fecha_entrada": datetime(2024, 1, 2),
            }
        return None


class TestBrokerInterface(unittest.TestCase):
    """Tests de BrokerInterface y enums."""

    def test_import(self):
        from ypyquant.execution import BrokerInterface, OrderSide, OrderStatus, OrderType  # noqa: F401
        self.assertEqual(OrderSide.BUY.value, "compra")
        self.assertEqual(OrderSide.SELL.value, "venta")
        self.assertEqual(OrderType.MARKET.value, "market")
        self.assertEqual(OrderStatus.FILLED.value, "filled")

    def test_broker_interface_is_abstract(self):
        from ypyquant.execution import BrokerInterface
        with self.assertRaises(TypeError):
            BrokerInterface({})  # No se puede instanciar directamente


class TestPaperBroker(unittest.TestCase):
    """Tests del PaperBroker (simulación)."""

    def setUp(self):
        from ypyquant.execution import PaperBroker
        self.broker = PaperBroker({
            "initial_cash": 10000.0,
            "currency": "USD",
            "commission": 0.001,
        })
        self.broker.connect()

    def test_connect(self):
        self.assertTrue(self.broker.connected)

    def test_account_info(self):
        info = self.broker.get_account_info()
        self.assertEqual(info["cash"], 10000.0)
        self.assertEqual(info["currency"], "USD")
        self.assertEqual(info["positions_count"], 0)

    def test_buy_increases_position(self):
        from ypyquant.execution import OrderSide, OrderStatus, OrderType  # noqa: F401
        # Simular precio inyectando directamente
        original_price = self.broker.get_current_price
        self.broker.get_current_price = lambda ticker: 100.0

        result = self.broker.place_order("AAPL", 10, OrderSide.BUY)
        self.assertEqual(result["status"], OrderStatus.FILLED)
        self.assertIn("AAPL", self.broker.positions)
        self.assertEqual(self.broker.positions["AAPL"]["quantity"], 10.0)

        self.broker.get_current_price = original_price

    def test_sell_reduces_position(self):
        from ypyquant.execution import OrderSide, OrderStatus
        self.broker.get_current_price = lambda ticker: 100.0

        self.broker.place_order("AAPL", 10, OrderSide.BUY)
        result = self.broker.place_order("AAPL", 5, OrderSide.SELL)

        self.assertEqual(result["status"], OrderStatus.FILLED)
        self.assertIn("AAPL", self.broker.positions)
        self.assertAlmostEqual(self.broker.positions["AAPL"]["quantity"], 5.0, places=4)

    def test_sell_all_closes_position(self):
        from ypyquant.execution import OrderSide
        self.broker.get_current_price = lambda ticker: 100.0

        self.broker.place_order("AAPL", 10, OrderSide.BUY)
        self.broker.place_order("AAPL", 10, OrderSide.SELL)

        self.assertNotIn("AAPL", self.broker.positions)

    def test_insufficient_funds_raises(self):
        from ypyquant.execution import OrderSide, OrderStatus
        self.broker.get_current_price = lambda ticker: 10000.0

        # Intentar comprar más de lo que hay
        result = self.broker.place_order("AAPL", 2, OrderSide.BUY)
        self.assertEqual(result["status"], OrderStatus.REJECTED)

    def test_reset(self):
        from ypyquant.execution import OrderSide
        self.broker.get_current_price = lambda ticker: 100.0
        self.broker.place_order("AAPL", 5, OrderSide.BUY)
        self.broker.reset()
        self.assertEqual(self.broker.cash, 10000.0)
        self.assertEqual(len(self.broker.positions), 0)

    def test_is_market_open_returns_bool(self):
        result = self.broker.is_market_open()
        self.assertIsInstance(result, bool)


class TestProtectionManager(unittest.TestCase):
    """Tests del ProtectionManager."""

    def test_import(self):
        from ypyquant.execution import (
            ProtectionManager,
            create_aggressive_protections,  # noqa: F401
            create_conservative_protections,
            create_moderate_protections,  # noqa: F401
        )
        protections = create_conservative_protections()
        manager = ProtectionManager(protections)
        self.assertIsNotNone(manager)

    def test_can_trade_initially(self):
        from ypyquant.execution import ProtectionManager, create_moderate_protections
        manager = ProtectionManager(create_moderate_protections())
        can_trade, reason = manager.check_can_trade(10000, 0)
        self.assertTrue(can_trade)
        self.assertIsNone(reason)

    def test_max_drawdown_blocks_trading(self):
        from ypyquant.execution import ProtectionConfig, ProtectionManager, ProtectionType
        manager = ProtectionManager([
            ProtectionConfig(ProtectionType.MAX_DRAWDOWN, max_drawdown_pct=0.10)
        ])
        manager.update_equity(10000)
        can_trade, reason = manager.check_can_trade(8500, 0)  # 15% drawdown
        self.assertFalse(can_trade)
        self.assertIn("Drawdown", reason)

    def test_position_limit_blocks_trading(self):
        from ypyquant.execution import ProtectionConfig, ProtectionManager, ProtectionType
        manager = ProtectionManager([
            ProtectionConfig(ProtectionType.POSITION_LIMIT, max_positions=3)
        ])
        can_trade, reason = manager.check_can_trade(10000, 3)
        self.assertFalse(can_trade)
        self.assertIn("Límite", reason)

    def test_get_status(self):
        from ypyquant.execution import ProtectionManager, create_conservative_protections
        manager = ProtectionManager(create_conservative_protections())
        status = manager.get_status()
        self.assertIn("active_protections", status)
        self.assertIn("metrics", status)
        self.assertFalse(status["is_blocked"])


class TestWalletManager(unittest.TestCase):
    """Tests del WalletManager."""

    def setUp(self):
        from ypyquant.execution import WalletManager
        self.wallet = WalletManager(
            initial_capital=10000.0,
            tradable_balance_ratio=0.99,
            max_open_trades=5,
            max_position_size=0.30,
        )

    def test_initial_balance(self):
        balance = self.wallet.get_balance()
        self.assertEqual(balance.total, 10000.0)
        self.assertEqual(balance.free, 10000.0)
        self.assertEqual(balance.used, 0.0)

    def test_tradable_balance(self):
        tradable = self.wallet.get_tradable_balance()
        self.assertAlmostEqual(tradable, 9900.0, places=1)

    def test_validate_stake_ok(self):
        valid, _msg = self.wallet.validate_stake_amount(1000.0, min_stake=10.0)
        self.assertTrue(valid)

    def test_validate_stake_exceeds_max_position(self):
        valid, msg = self.wallet.validate_stake_amount(4000.0, min_stake=10.0)
        self.assertFalse(valid)
        self.assertIn("máximo", msg)

    def test_validate_stake_too_many_trades(self):
        valid, msg = self.wallet.validate_stake_amount(100.0, min_stake=10.0, current_open_trades=5)
        self.assertFalse(valid)
        self.assertIn("trades", msg)

    def test_calculate_position_size(self):
        qty, cost = self.wallet.calculate_position_size(1000.0, price=100.0)
        self.assertGreater(qty, 0)
        self.assertEqual(cost, 1000.0)

    def test_invalid_ratios_raise(self):
        from ypyquant.execution import WalletManager
        with self.assertRaises(ValueError):
            WalletManager(10000.0, tradable_balance_ratio=1.5)
        with self.assertRaises(ValueError):
            WalletManager(10000.0, max_position_size=0.0)


class TestTradeAnalytics(unittest.TestCase):
    """Tests de TradeAnalytics (métodos estáticos + con storage)."""

    def test_calculate_trade_pnl_long(self):
        from ypyquant.execution import TradeAnalytics
        pnl, pct = TradeAnalytics.calculate_trade_pnl(100.0, 120.0, 10)
        self.assertAlmostEqual(pnl, 200.0, places=2)
        self.assertGreater(pct, 0)

    def test_calculate_trade_pnl_loss(self):
        from ypyquant.execution import TradeAnalytics
        pnl, pct = TradeAnalytics.calculate_trade_pnl(100.0, 80.0, 10)
        self.assertLess(pnl, 0)
        self.assertLess(pct, 0)

    def test_calculate_trade_pnl_short(self):
        from ypyquant.execution import TradeAnalytics
        pnl, _pct = TradeAnalytics.calculate_trade_pnl(100.0, 80.0, 10, is_short=True)
        self.assertGreater(pnl, 0)

    def test_calculate_position_metrics(self):
        from ypyquant.execution import TradeAnalytics
        metrics = TradeAnalytics.calculate_position_metrics(
            ticker="AAPL",
            entry_price=150.0,
            current_price=160.0,
            quantity=10.0,
            entry_fee=1.5,
            entry_date=datetime(2024, 1, 2),
        )
        self.assertEqual(metrics.ticker, "AAPL")
        self.assertTrue(metrics.is_open)
        self.assertGreater(metrics.unrealized_pnl, 0)

    def test_portfolio_metrics_with_storage(self):
        from ypyquant.execution import TradeAnalytics
        storage = MockStorage()
        analytics = TradeAnalytics(storage=storage)
        metrics = analytics.calculate_portfolio_metrics(portafolio_id=1)
        self.assertGreaterEqual(metrics.total_trades, 0)

    def test_get_position_metrics(self):
        from ypyquant.execution import TradeAnalytics
        storage = MockStorage()
        analytics = TradeAnalytics(storage=storage)
        posicion = storage.get_posicion(1, "AAPL")
        metrics = analytics.get_position_metrics(posicion, current_price=165.0)
        self.assertEqual(metrics.ticker, "AAPL")
        self.assertTrue(metrics.is_open)

    def test_format_trade_summary(self):
        from ypyquant.execution import TradeAnalytics
        metrics = TradeAnalytics.calculate_position_metrics(
            ticker="MSFT", entry_price=300.0, current_price=310.0,
            quantity=5.0, entry_fee=0.3, entry_date=datetime(2024, 1, 5),
        )
        summary = TradeAnalytics.format_trade_summary(metrics)
        self.assertIn("MSFT", summary)
        self.assertIn("ABIERTO", summary)


class TestBotExecutor(unittest.TestCase):
    """Tests de BotExecutor con mock storage y PaperBroker."""

    def setUp(self):
        from ypyquant.execution import BotExecutor, PaperBroker
        self.storage = MockStorage()
        self.broker = PaperBroker({
            "initial_cash": 10000.0,
            "currency": "USD",
            "commission": 0.001,
        })
        # Inyectar precio fijo para evitar llamadas a yfinance
        self.broker.get_current_price = lambda ticker: 150.0

        self.executor = BotExecutor(
            storage=self.storage,
            portafolio_id=1,
            broker=self.broker,
        )

    def test_init(self):
        self.assertEqual(self.executor.portafolio_id, 1)
        self.assertEqual(self.executor.portafolio["nombre"], "Test Portfolio")

    def test_get_portfolio_status(self):
        status = self.executor.get_portfolio_status()
        self.assertIn("portafolio_id", status)
        self.assertIn("account_info", status)
        self.assertIn("positions", status)

    def test_execute_rebalance_dry_run(self):
        # Usar pesos pequeños para no agotar el capital con comisiones
        result = self.executor.execute_rebalance(
            target_weights={"AAPL": 0.45, "MSFT": 0.45, "GOOGL": 0.10},
            dry_run=True,
        )
        self.assertIn("success", result)
        # Si es factible, debe estar marcado como dry_run
        if result["success"]:
            self.assertTrue(result.get("dry_run"))

    def test_get_wallet_status(self):
        status = self.executor.get_wallet_status()
        self.assertIn("balance", status)
        self.assertIn("limits", status)
        self.assertIn("capacity", status)

    def test_get_operation_history(self):
        history = self.executor.get_operation_history()
        self.assertIsInstance(history, list)

    def test_close(self):
        self.executor.close()
        self.assertFalse(self.broker.connected)


class TestPublicAPI(unittest.TestCase):
    """Verifica que los símbolos públicos son importables desde ypyquant directamente."""

    def test_top_level_imports(self):
        from ypyquant import (
            BotExecutor,
            BrokerInterface,  # noqa: F401
            OrderSide,  # noqa: F401
            OrderStatus,  # noqa: F401
            OrderType,  # noqa: F401
            PaperBroker,
            ProtectionManager,
            StorageInterface,  # noqa: F401
            TradeAnalytics,  # noqa: F401
            WalletManager,
        )
        # Todos deben ser clases/enums
        self.assertTrue(callable(PaperBroker))
        self.assertTrue(callable(ProtectionManager))
        self.assertTrue(callable(WalletManager))
        self.assertTrue(callable(BotExecutor))
