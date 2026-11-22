"""
ypyquant.execution — ejecución: calendario, Ejecutor (pesos -> órdenes), puerto de broker y utilidades.

Importaciones públicas:
    from ypyquant.execution import planificar_rebalanceo, PoliticaEjecucion, POLITICA_MOTOR, POLITICA_BOT
    from ypyquant.execution import toca_rebalancear, fechas_de_rebalanceo, FRECUENCIAS_VALIDAS
    from ypyquant.execution import Broker, BrokerSimulado       (BrokerAlpaca: ypyquant.execution.broker_alpaca)
    from ypyquant.execution import ProtectionManager, TradeAnalytics, StorageInterface

Se eliminaron (v0.6, ver docs/planes/2026-11-21-ejecutor.md) BotExecutor, PaperBroker, BrokerInterface,
WalletManager y los enums OrderSide/OrderType/OrderStatus, heredados de GrowSave y sin uso (GrowSave solo
importa TradeAnalytics y StorageInterface): los sustituyen el Ejecutor y el puerto `Broker`.
"""

from ypyquant.execution.broker import ESTADOS_TERMINALES, Broker, EstadoOrden
from ypyquant.execution.broker_simulado import BrokerSimulado
from ypyquant.execution.calendario import (
    FRECUENCIAS_VALIDAS,
    fechas_de_rebalanceo,
    toca_rebalancear,
)
from ypyquant.execution.ejecutor import (
    POLITICA_BOT,
    POLITICA_MOTOR,
    TICKER_EFECTIVO,
    EstadoCuenta,
    Orden,
    PlanRebalanceo,
    PoliticaEjecucion,
    planificar_rebalanceo,
)
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

__all__ = [
    "ESTADOS_TERMINALES",
    "FRECUENCIAS_VALIDAS",
    "POLITICA_BOT",
    "POLITICA_MOTOR",
    "TICKER_EFECTIVO",
    "Broker",
    "BrokerSimulado",
    "EstadoCuenta",
    "EstadoOrden",
    "Orden",
    "PlanRebalanceo",
    "PoliticaEjecucion",
    "PortfolioMetrics",
    "ProtectionConfig",
    "ProtectionManager",
    "ProtectionType",
    "StorageInterface",
    "TradeAnalytics",
    "TradeMetrics",
    "create_aggressive_protections",
    "create_conservative_protections",
    "create_moderate_protections",
    "fechas_de_rebalanceo",
    "planificar_rebalanceo",
    "toca_rebalancear",
]
