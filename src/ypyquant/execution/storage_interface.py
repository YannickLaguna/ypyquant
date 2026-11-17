"""
StorageInterface — abstracción de persistencia para ypyquant.execution.

Permite desacoplar BotExecutor y TradeAnalytics de cualquier ORM concreto.
TFM implementa esta interfaz usando SQLAlchemy (ver app/services/quant_bridge.py).
"""

from abc import ABC, abstractmethod
from datetime import datetime


class StorageInterface(ABC):
    """
    Interfaz abstracta de persistencia.

    El consumidor (TFM) implementa esta clase usando su ORM propio.
    ypyquant solo depende de esta abstracción, nunca de SQLAlchemy ni de modelos TFM.
    """

    @abstractmethod
    def load_portafolio(self, portafolio_id: int) -> dict:
        """
        Carga un portafolio por ID.

        Returns:
            Dict con: nombre, type, capital_actual, target_weights, rebalance_threshold
        Raises:
            ValueError: Si el portafolio no existe
        """
        ...

    @abstractmethod
    def update_portafolio(self, portafolio_id: int, capital_actual: float, last_optimization_date: datetime):
        """Actualiza capital_actual y last_optimization_date del portafolio."""
        ...

    @abstractmethod
    def save_operation(self, portafolio_id: int, order_info: dict) -> int:
        """
        Persiste una operación ejecutada.

        Args:
            portafolio_id: ID del portafolio
            order_info: Dict con ticker, side, filled_quantity, filled_price, filled_at, commission

        Returns:
            ID de la operación creada en BD
        """
        ...

    @abstractmethod
    def sync_positions(self, portafolio_id: int, positions: list[dict]):
        """
        Sincroniza posiciones en BD.
        Elimina posiciones antiguas del portafolio e inserta las nuevas.

        Args:
            portafolio_id: ID del portafolio
            positions: Lista de dicts con ticker, quantity, avg_entry_price, current_price
        """
        ...

    @abstractmethod
    def get_operations(
        self,
        portafolio_id: int,
        ticker: str | None = None,
        limit: int = 0,
    ) -> list[dict]:
        """
        Obtiene operaciones de un portafolio.

        Returns:
            Lista de dicts con: id, ticker, tipo ('compra'|'venta'), cantidad, precio, comision, fecha
        """
        ...

    @abstractmethod
    def get_posicion(self, portafolio_id: int, ticker: str) -> dict | None:
        """
        Obtiene una posición específica.

        Returns:
            Dict con: portafolio_id, ticker, precio_entrada, cantidad, fecha_entrada | None
        """
        ...
