"""
Tests de smoke para ypyquant.execution (M0, Fase 3+4).

Responsabilidad:
    Verificar la migración de TFM a ypyquant.execution: ProtectionManager,
    TradeAnalytics y API pública. (BrokerInterface, PaperBroker, WalletManager y BotExecutor se eliminaron: ver
    docs/planes/2026-11-21-ejecutor.md; el Ejecutor y el puerto Broker se prueban en tests/test_ejecutor*.py.) Usa un
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


class TestPublicAPI(unittest.TestCase):
    """Verifica que los símbolos públicos son importables desde ypyquant directamente."""

    def test_top_level_imports(self):
        from ypyquant import (
            Broker,
            BrokerSimulado,
            PoliticaEjecucion,
            ProtectionManager,
            StorageInterface,  # noqa: F401
            TradeAnalytics,  # noqa: F401
            planificar_rebalanceo,
        )
        self.assertTrue(callable(BrokerSimulado))
        self.assertTrue(callable(ProtectionManager))
        self.assertTrue(callable(PoliticaEjecucion))
        self.assertTrue(callable(planificar_rebalanceo))
        self.assertTrue(isinstance(BrokerSimulado(), Broker))  # el simulado cumple el puerto
