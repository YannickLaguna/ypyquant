"""
Gestión inteligente de capital y validación de operaciones.
Migrado de tfm/app/services/trading/wallet_manager.py

Responsabilidades:
- Calcular capital disponible para trading
- Validar que hay suficiente capital antes de operar
- Respetar límites configurables (max_position_size, reserve_amount)
- Calcular tamaño óptimo de posición
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WalletBalance:
    total: float
    free: float
    used: float


class WalletManager:
    """
    Gestor de capital para portafolios.
    Controla cuánto capital está disponible, valida operaciones, calcula tamaños de posición.
    """

    def __init__(
        self,
        initial_capital: float,
        tradable_balance_ratio: float = 0.99,
        max_open_trades: int = 10,
        stake_amount: Optional[float] = None,
        reserve_amount: float = 0.0,
        max_position_size: float = 1.0,
    ):
        if not 0 < tradable_balance_ratio <= 1.0:
            raise ValueError("tradable_balance_ratio debe estar entre 0 y 1")
        if not 0 < max_position_size <= 1.0:
            raise ValueError("max_position_size debe estar entre 0 y 1")

        self.initial_capital = initial_capital
        self.tradable_balance_ratio = tradable_balance_ratio
        self.max_open_trades = max_open_trades
        self.stake_amount = stake_amount
        self.reserve_amount = reserve_amount
        self.max_position_size = max_position_size

        self._total_capital = initial_capital
        self._free_capital = initial_capital
        self._used_capital = 0.0

        logger.info(
            f"WalletManager — Capital: ${initial_capital:,.2f}, "
            f"Tradable: {tradable_balance_ratio:.0%}, Reserve: ${reserve_amount:,.2f}, "
            f"Max position: {max_position_size:.0%}"
        )

    def update_balance(self, total_capital: float, used_capital: float = 0.0):
        self._total_capital = total_capital
        self._used_capital = used_capital
        self._free_capital = total_capital - used_capital

    def get_balance(self) -> WalletBalance:
        return WalletBalance(
            total=self._total_capital,
            free=self._free_capital,
            used=self._used_capital,
        )

    def get_tradable_balance(self) -> float:
        tradable = self._free_capital * self.tradable_balance_ratio
        return max(0, tradable - self.reserve_amount)

    def get_trade_stake_amount(
        self,
        current_open_trades: int = 0,
        custom_stake: Optional[float] = None,
    ) -> float:
        if custom_stake is not None:
            stake = custom_stake
        elif self.stake_amount is not None:
            stake = self.stake_amount
        else:
            tradable = self.get_tradable_balance()
            remaining_slots = max(1, self.max_open_trades - current_open_trades)
            stake = tradable / remaining_slots

        max_stake = self._total_capital * self.max_position_size
        return max(0, min(stake, max_stake))

    def validate_stake_amount(
        self,
        stake_amount: float,
        min_stake: float = 10.0,
        current_open_trades: int = 0,
    ) -> Tuple[bool, str]:
        if stake_amount < min_stake:
            return False, f"Stake ${stake_amount:.2f} menor al mínimo ${min_stake:.2f}"

        max_stake = self._total_capital * self.max_position_size
        if stake_amount > max_stake:
            return False, (
                f"Stake ${stake_amount:.2f} excede máximo ${max_stake:.2f} "
                f"({self.max_position_size:.0%} del portafolio)"
            )

        tradable = self.get_tradable_balance()
        if stake_amount > tradable:
            return False, f"Capital insuficiente. Stake: ${stake_amount:.2f}, Disponible: ${tradable:.2f}"

        if current_open_trades >= self.max_open_trades:
            return False, f"Máximo de trades alcanzado ({current_open_trades}/{self.max_open_trades})"

        return True, ""

    def can_execute_trade(
        self,
        stake_amount: float,
        min_stake: float = 10.0,
        current_open_trades: int = 0,
    ) -> bool:
        is_valid, _ = self.validate_stake_amount(stake_amount, min_stake, current_open_trades)
        return is_valid

    def calculate_position_size(
        self,
        stake_amount: float,
        price: float,
        commission_rate: float = 0.001,
    ) -> Tuple[float, float]:
        if price <= 0:
            raise ValueError("El precio debe ser mayor a 0")
        commission = stake_amount * commission_rate
        quantity = (stake_amount - commission) / price
        return quantity, stake_amount

    def get_available_stake_for_ticker(
        self,
        ticker: str,
        current_position_value: float = 0.0,
    ) -> float:
        max_allowed = self._total_capital * self.max_position_size
        available = max(0, max_allowed - current_position_value)
        return min(available, self.get_tradable_balance())

    def estimate_max_trades(self, avg_stake_per_trade: Optional[float] = None) -> int:
        tradable = self.get_tradable_balance()
        if avg_stake_per_trade is None:
            avg_stake_per_trade = (
                self.stake_amount
                if self.stake_amount is not None
                else tradable / max(1, self.max_open_trades)
            )
        if avg_stake_per_trade <= 0:
            return 0
        return min(int(tradable / avg_stake_per_trade), self.max_open_trades)

    def get_portfolio_allocation(self) -> Dict:
        balance = self.get_balance()
        tradable = self.get_tradable_balance()
        return {
            "total_capital": balance.total,
            "free_capital": balance.free,
            "used_capital": balance.used,
            "tradable_balance": tradable,
            "reserved_capital": self.reserve_amount,
            "allocation_ratio": balance.used / balance.total if balance.total > 0 else 0,
            "free_ratio": balance.free / balance.total if balance.total > 0 else 0,
            "max_position_value": balance.total * self.max_position_size,
        }

    def check_sufficient_funds(self, required_amount: float, operation: str = "operación") -> Tuple[bool, str]:
        tradable = self.get_tradable_balance()
        if required_amount <= tradable:
            return True, f"Fondos suficientes para {operation}"
        return False, (
            f"Fondos insuficientes para {operation}. "
            f"Requerido: ${required_amount:,.2f}, Disponible: ${tradable:,.2f}"
        )

    def __repr__(self) -> str:
        b = self.get_balance()
        return f"WalletManager(total=${b.total:,.2f}, free=${b.free:,.2f}, used=${b.used:,.2f})"
