"""
ypyquant.execution — módulo de ejecución de trading.
Migrado de tfm/app/services/trading/ en Fase 3+4.

Importaciones públicas:
    from ypyquant.execution import BrokerInterface, PaperBroker, BotExecutor
    from ypyquant.execution import ProtectionManager, WalletManager, TradeAnalytics
    from ypyquant.execution import StorageInterface
    from ypyquant.execution import OrderSide, OrderType, OrderStatus
"""

from ypyquant.execution.bot_executor import BotExecutor
from ypyquant.execution.broker_interface import (
    BrokerInterface,
    OrderSide,
    OrderStatus,
    OrderType,
)
from ypyquant.execution.paper_broker import PaperBroker
from ypyquant.execution.protection_manager import (
    ProtectionConfig,
    ProtectionManager,
    ProtectionType,
    create_aggressive_protections,
    create_conservative_protections,
    create_moderate_protections,
)
from ypyquant.execution.storage_interface import StorageInterface
from ypyquant.execution.trade_analytics import PortfolioMetrics, TradeAnalytics, TradeMetrics
from ypyquant.execution.wallet_manager import WalletBalance, WalletManager

__all__ = [
    # Broker
    "BrokerInterface",
    "PaperBroker",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    # Protection
    "ProtectionManager",
    "ProtectionConfig",
    "ProtectionType",
    "create_conservative_protections",
    "create_moderate_protections",
    "create_aggressive_protections",
    # Wallet
    "WalletManager",
    "WalletBalance",
    # Analytics
    "TradeAnalytics",
    "TradeMetrics",
    "PortfolioMetrics",
    # Storage
    "StorageInterface",
    # Executor
    "BotExecutor",
]
