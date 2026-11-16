"""
Protection Manager — Circuit Breaker y Gestión de Riesgo.
Migrado de tfm/app/services/trading/protection_manager.py

Responsabilidades:
- Detener trading en condiciones adversas
- Circuit breakers basados en pérdidas
- Cooldown periods después de drawdowns
- Límites de pérdidas diarias/semanales
- Bloqueo de activos específicos
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProtectionType(Enum):
    MAX_DRAWDOWN = "max_drawdown"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    COOLDOWN_PERIOD = "cooldown_period"
    POSITION_LIMIT = "position_limit"
    VOLATILITY_GUARD = "volatility_guard"


class ProtectionStatus(Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


@dataclass
class ProtectionConfig:
    protection_type: ProtectionType
    enabled: bool = True

    max_drawdown_pct: Optional[float] = None
    daily_loss_limit: Optional[float] = None
    max_consecutive_losses: Optional[int] = None
    cooldown_minutes: Optional[int] = None
    max_positions: Optional[int] = None
    max_volatility_pct: Optional[float] = None


@dataclass
class ProtectionEvent:
    protection_type: ProtectionType
    triggered_at: datetime
    reason: str
    expires_at: Optional[datetime] = None
    data: Optional[Dict] = None


class ProtectionManager:
    """
    Gestor de protecciones y circuit breakers.
    Previene operaciones riesgosas y detiene el trading cuando se alcanzan límites de riesgo.
    """

    def __init__(self, protections: List[ProtectionConfig]):
        self.protections = {p.protection_type: p for p in protections if p.enabled}
        self.active_protections: List[ProtectionEvent] = []

        self._peak_equity = 0.0
        self._daily_loss_tracker: Dict = {}
        self._consecutive_losses = 0
        self._last_trade_timestamp = None

        logger.info(f"ProtectionManager inicializado con {len(self.protections)} protecciones")

    def check_can_trade(
        self,
        current_equity: float,
        current_positions: int,
        ticker: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        self._cleanup_expired_protections()

        if self.active_protections:
            protection = self.active_protections[0]
            return False, f"Trading bloqueado: {protection.reason}"

        for ptype, config in self.protections.items():
            if ptype == ProtectionType.MAX_DRAWDOWN:
                can_trade, reason = self._check_max_drawdown(current_equity, config)
                if not can_trade:
                    return False, reason

            elif ptype == ProtectionType.DAILY_LOSS_LIMIT:
                can_trade, reason = self._check_daily_loss_limit(config)
                if not can_trade:
                    return False, reason

            elif ptype == ProtectionType.CONSECUTIVE_LOSSES:
                can_trade, reason = self._check_consecutive_losses(config)
                if not can_trade:
                    return False, reason

            elif ptype == ProtectionType.COOLDOWN_PERIOD:
                can_trade, reason = self._check_cooldown_period(config)
                if not can_trade:
                    return False, reason

            elif ptype == ProtectionType.POSITION_LIMIT:
                can_trade, reason = self._check_position_limit(current_positions, config)
                if not can_trade:
                    return False, reason

        return True, None

    def _check_max_drawdown(self, current_equity, config) -> Tuple[bool, Optional[str]]:
        if config.max_drawdown_pct is None:
            return True, None
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= config.max_drawdown_pct:
                reason = (
                    f"Drawdown máximo alcanzado: {drawdown:.1%} "
                    f"(límite: {config.max_drawdown_pct:.1%})"
                )
                self._trigger_protection(
                    ProtectionType.MAX_DRAWDOWN, reason,
                    {"drawdown": drawdown, "peak_equity": self._peak_equity},
                )
                return False, reason
        return True, None

    def _check_daily_loss_limit(self, config) -> Tuple[bool, Optional[str]]:
        if config.daily_loss_limit is None:
            return True, None
        today = datetime.now().date()
        daily_loss = self._daily_loss_tracker.get(today, 0.0)
        if abs(daily_loss) >= config.daily_loss_limit:
            reason = (
                f"Límite de pérdida diaria alcanzado: ${abs(daily_loss):,.2f} "
                f"(límite: ${config.daily_loss_limit:,.2f})"
            )
            expires_at = datetime.combine(today + timedelta(days=1), datetime.min.time())
            self._trigger_protection(
                ProtectionType.DAILY_LOSS_LIMIT, reason, {"daily_loss": daily_loss}, expires_at
            )
            return False, reason
        return True, None

    def _check_consecutive_losses(self, config) -> Tuple[bool, Optional[str]]:
        if config.max_consecutive_losses is None:
            return True, None
        if self._consecutive_losses >= config.max_consecutive_losses:
            reason = (
                f"Pérdidas consecutivas alcanzadas: {self._consecutive_losses} "
                f"(límite: {config.max_consecutive_losses})"
            )
            expires_at = datetime.now() + timedelta(hours=1)
            self._trigger_protection(
                ProtectionType.CONSECUTIVE_LOSSES, reason,
                {"consecutive_losses": self._consecutive_losses}, expires_at,
            )
            return False, reason
        return True, None

    def _check_cooldown_period(self, config) -> Tuple[bool, Optional[str]]:
        if config.cooldown_minutes is None or self._last_trade_timestamp is None:
            return True, None
        time_since = (datetime.now() - self._last_trade_timestamp).total_seconds() / 60
        if time_since < config.cooldown_minutes:
            remaining = config.cooldown_minutes - time_since
            return False, f"Período de enfriamiento: {remaining:.1f} minutos restantes"
        return True, None

    def _check_position_limit(self, current_positions, config) -> Tuple[bool, Optional[str]]:
        if config.max_positions is None:
            return True, None
        if current_positions >= config.max_positions:
            reason = (
                f"Límite de posiciones alcanzado: {current_positions} "
                f"(límite: {config.max_positions})"
            )
            return False, reason
        return True, None

    def _trigger_protection(self, protection_type, reason, data=None, expires_at=None):
        event = ProtectionEvent(
            protection_type=protection_type,
            triggered_at=datetime.now(),
            reason=reason,
            expires_at=expires_at,
            data=data,
        )
        self.active_protections.append(event)
        logger.warning(f"PROTECCIÓN DISPARADA: {reason}")

    def _cleanup_expired_protections(self):
        now = datetime.now()
        self.active_protections = [
            p for p in self.active_protections
            if not (p.expires_at and p.expires_at <= now)
        ]

    def update_equity(self, current_equity: float):
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    def record_trade_result(self, pnl: float):
        today = datetime.now().date()
        self._daily_loss_tracker.setdefault(today, 0.0)
        self._daily_loss_tracker[today] += pnl

        cutoff = today - timedelta(days=7)
        self._daily_loss_tracker = {
            d: loss for d, loss in self._daily_loss_tracker.items() if d >= cutoff
        }

        self._consecutive_losses = self._consecutive_losses + 1 if pnl < 0 else 0
        self._last_trade_timestamp = datetime.now()

    def get_status(self) -> Dict:
        self._cleanup_expired_protections()
        today = datetime.now().date()
        return {
            "active_protections": [
                {
                    "type": p.protection_type.value,
                    "triggered_at": p.triggered_at.isoformat(),
                    "reason": p.reason,
                    "expires_at": p.expires_at.isoformat() if p.expires_at else None,
                }
                for p in self.active_protections
            ],
            "metrics": {
                "peak_equity": self._peak_equity,
                "daily_loss": self._daily_loss_tracker.get(today, 0.0),
                "consecutive_losses": self._consecutive_losses,
                "last_trade": self._last_trade_timestamp.isoformat() if self._last_trade_timestamp else None,
            },
            "is_blocked": len(self.active_protections) > 0,
        }

    def reset(self):
        self.active_protections.clear()
        self._peak_equity = 0.0
        self._daily_loss_tracker.clear()
        self._consecutive_losses = 0
        self._last_trade_timestamp = None
        logger.info("ProtectionManager reseteado")

    def override_protection(self, protection_type: ProtectionType):
        self.active_protections = [
            p for p in self.active_protections if p.protection_type != protection_type
        ]
        logger.warning(f"Protección {protection_type.value} sobrescrita manualmente")


def create_conservative_protections() -> List[ProtectionConfig]:
    return [
        ProtectionConfig(ProtectionType.MAX_DRAWDOWN, max_drawdown_pct=0.10),
        ProtectionConfig(ProtectionType.DAILY_LOSS_LIMIT, daily_loss_limit=500.0),
        ProtectionConfig(ProtectionType.CONSECUTIVE_LOSSES, max_consecutive_losses=3),
        ProtectionConfig(ProtectionType.POSITION_LIMIT, max_positions=5),
        ProtectionConfig(ProtectionType.COOLDOWN_PERIOD, cooldown_minutes=30),
    ]


def create_moderate_protections() -> List[ProtectionConfig]:
    return [
        ProtectionConfig(ProtectionType.MAX_DRAWDOWN, max_drawdown_pct=0.20),
        ProtectionConfig(ProtectionType.DAILY_LOSS_LIMIT, daily_loss_limit=1000.0),
        ProtectionConfig(ProtectionType.CONSECUTIVE_LOSSES, max_consecutive_losses=5),
        ProtectionConfig(ProtectionType.POSITION_LIMIT, max_positions=10),
    ]


def create_aggressive_protections() -> List[ProtectionConfig]:
    return [
        ProtectionConfig(ProtectionType.MAX_DRAWDOWN, max_drawdown_pct=0.30),
        ProtectionConfig(ProtectionType.DAILY_LOSS_LIMIT, daily_loss_limit=2000.0),
        ProtectionConfig(ProtectionType.POSITION_LIMIT, max_positions=20),
    ]
