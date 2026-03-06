"""
Servicio de rebalanceo de carteras.
Calcula las operaciones necesarias para ajustar una cartera actual a una cartera objetivo.
Diseñado como microservicio independiente sin dependencias del broker.
"""

import logging
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Tipo de operación de rebalanceo"""
    BUY = "compra"
    SELL = "venta"
    HOLD = "mantener"


@dataclass
class Position:
    """Representa una posición en la cartera"""
    ticker: str
    quantity: float
    price: float  # Precio actual de mercado

    @property
    def market_value(self) -> float:
        """Valor de mercado de la posición"""
        return self.quantity * self.price


@dataclass
class RebalanceOperation:
    """Representa una operación de rebalanceo planificada"""
    ticker: str
    operation_type: OperationType
    quantity: float  # Cantidad de acciones a comprar/vender
    current_price: float
    trade_value: float  # Valor bruto de la operación (sin signo)
    commission: float  # Comisión aplicada a la operación
    estimated_cost: float  # Costo/ingreso neto incluyendo comisión

    # Información adicional para análisis
    current_weight: float
    target_weight: float
    weight_diff: float
    reason: str


class RebalanceService:
    """
    Servicio de rebalanceo de carteras.

    Calcula las operaciones necesarias para transformar una cartera actual
    en una cartera objetivo, considerando umbrales y restricciones.
    """

    def __init__(
        self,
        rebalance_threshold: float = 0.05,  # 5% por defecto
        min_trade_value: float = 100.0,     # Mínimo $100 por operación
        commission_rate: float = 0.001,     # 0.1% comisión
        price_fetcher: Optional[Callable[[List[str]], Dict[str, float]]] = None
    ):
        """
        Inicializa el servicio de rebalanceo.

        Args:
            rebalance_threshold: Diferencia mínima de peso para operar (ej: 0.05 = 5%)
            min_trade_value: Valor mínimo en $ para ejecutar una operación
            commission_rate: Tasa de comisión por operación
            price_fetcher: Función opcional para obtener precios actuales dado un listado de tickers
        """
        self.rebalance_threshold = rebalance_threshold
        self.min_trade_value = min_trade_value
        self.commission_rate = commission_rate
        self.price_fetcher = price_fetcher

        logger.info(
            f"RebalanceService inicializado - "
            f"Umbral: {rebalance_threshold:.1%}, "
            f"Min trade: ${min_trade_value}, "
            f"Comisión: {commission_rate:.2%}"
        )

    def calculate_rebalance(
        self,
        current_positions: List[Position],
        target_weights: Dict[str, float],
        available_cash: float = 0.0,
        current_prices: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Calcula las operaciones necesarias para rebalancear la cartera.

        Args:
            current_positions: Lista de posiciones actuales
            target_weights: Pesos objetivo por ticker {'AAPL': 0.3, 'GOOGL': 0.4, ...}
            available_cash: Cash disponible para inversión
            current_prices: Precios actuales por ticker (opcional, se usa Position.price si no se provee)

        Returns:
            Dict con:
                - operations: List[RebalanceOperation] - Operaciones planificadas
                - summary: Dict - Resumen del rebalanceo
                - current_allocation: Dict - Allocación actual
                - target_allocation: Dict - Allocación objetivo
        """
        logger.info("=" * 70)
        logger.info("Calculando rebalanceo de cartera")
        logger.info("=" * 70)

        # Validar target_weights
        if not self._validate_target_weights(target_weights):
            raise ValueError("Target weights inválidos (deben sumar 1.0)")

        # Calcular valor total del portafolio
        total_portfolio_value = self._calculate_total_value(current_positions, available_cash)

        logger.info(f"Valor total del portafolio: ${total_portfolio_value:,.2f}")
        logger.info(f"  - Posiciones: ${sum(p.market_value for p in current_positions):,.2f}")
        logger.info(f"  - Cash disponible: ${available_cash:,.2f}")

        # Calcular pesos actuales
        current_weights = self._calculate_current_weights(current_positions, total_portfolio_value)

        logger.info(f"\nPesos actuales: {self._format_weights(current_weights)}")
        logger.info(f"Pesos objetivo: {self._format_weights(target_weights)}")

        # Crear mapa de posiciones actuales
        positions_map = {pos.ticker: pos for pos in current_positions}

        # Usar precios del mapa si se proveen, sino usar los de Position; completar faltantes con price_fetcher
        resolved_prices = self._resolve_prices(target_weights, positions_map, current_prices)

        # Calcular operaciones necesarias
        operations = self._calculate_operations(
            target_weights,
            current_weights,
            total_portfolio_value,
            positions_map,
            resolved_prices
        )

        # Generar resumen
        summary = self._generate_summary(operations, current_weights, target_weights, total_portfolio_value)

        logger.info("\n" + "=" * 70)
        logger.info(f"Rebalanceo calculado: {len(operations)} operaciones planificadas")
        logger.info("=" * 70)

        return {
            'operations': operations,
            'summary': summary,
            'current_allocation': current_weights,
            'target_allocation': target_weights,
            'total_portfolio_value': total_portfolio_value
        }

    def _validate_target_weights(self, target_weights: Dict[str, float]) -> bool:
        """Valida que los pesos objetivo sumen 1.0 (con tolerancia)."""
        if not target_weights:
            return False

        total = sum(target_weights.values())

        # Tolerancia de 1%
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Target weights suman {total:.4f}, no 1.0")
            return False

        # Verificar que todos los pesos sean >= 0
        if any(w < 0 for w in target_weights.values()):
            logger.error("Target weights negativos detectados")
            return False

        return True

    def _calculate_total_value(self, positions: List[Position], cash: float) -> float:
        """Calcula el valor total del portafolio."""
        positions_value = sum(pos.market_value for pos in positions)
        return positions_value + cash

    def _calculate_current_weights(
        self,
        positions: List[Position],
        total_value: float
    ) -> Dict[str, float]:
        """Calcula los pesos actuales de la cartera."""
        if total_value == 0:
            return {}

        weights = {}
        for pos in positions:
            weights[pos.ticker] = pos.market_value / total_value

        return weights

    def _resolve_prices(
        self,
        target_weights: Dict[str, float],
        positions_map: Dict[str, Position],
        current_prices: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Resuelve precios actuales combinando entradas, posiciones y un fetcher opcional.

        Parameters
        ----------
        target_weights : Dict[str, float]
            Pesos objetivo por ticker.
        positions_map : Dict[str, Position]
            Posiciones actuales indexadas por ticker.
        current_prices : Optional[Dict[str, float]]
            Precios proporcionados externamente.

        Returns
        -------
        Dict[str, float]
            Precios actuales por ticker.

        Raises
        ------
        ValueError
            Si faltan precios y no se pueden obtener mediante price_fetcher.
        """
        prices: Dict[str, float] = {}

        # Empezar con precios provistos o de las posiciones
        if current_prices:
            prices.update(current_prices)
        else:
            prices.update({ticker: pos.price for ticker, pos in positions_map.items()})

        tickers_needed = set(target_weights.keys()) | set(positions_map.keys())
        missing = [t for t in tickers_needed if t not in prices]

        if missing and self.price_fetcher:
            fetched = self.price_fetcher(missing)
            prices.update(fetched or {})
            missing = [t for t in tickers_needed if t not in prices]

        if missing:
            raise ValueError(f"No hay precios disponibles para: {', '.join(sorted(missing))}")

        return prices

    def _calculate_operations(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        total_value: float,
        positions_map: Dict[str, Position],
        current_prices: Dict[str, float]
    ) -> List[RebalanceOperation]:
        """Calcula las operaciones necesarias para rebalancear."""
        operations = []

        # Obtener todos los tickers (actuales + objetivo)
        all_tickers = set(target_weights.keys()) | set(current_weights.keys())

        for ticker in all_tickers:
            target_weight = target_weights.get(ticker, 0.0)
            current_weight = current_weights.get(ticker, 0.0)

            weight_diff = target_weight - current_weight

            # Verificar umbral de rebalanceo
            if abs(weight_diff) < self.rebalance_threshold:
                logger.debug(
                    f"{ticker}: Diferencia {weight_diff:+.2%} < umbral {self.rebalance_threshold:.2%}, "
                    f"omitiendo"
                )
                continue

            # Obtener precio actual
            if ticker not in current_prices:
                logger.warning(f"{ticker}: No hay precio disponible, omitiendo")
                continue

            current_price = current_prices[ticker]

            # Calcular valores
            target_value = target_weight * total_value
            current_value = current_weight * total_value
            value_diff = target_value - current_value

            # Calcular cantidad a operar (en acciones)
            quantity = abs(value_diff) / current_price
            # TODO(opcional): redondear quantity a enteros si el broker no permite fracciones.

            # Verificar valor mínimo de operación
            operation_value = quantity * current_price
            if operation_value < self.min_trade_value:
                logger.debug(
                    f"{ticker}: Valor de operación ${operation_value:.2f} < mínimo ${self.min_trade_value}, "
                    f"omitiendo"
                )
                continue

            # Estimar costo (incluye comisión)
            commission = operation_value * self.commission_rate
            estimated_cost = operation_value + commission if value_diff > 0 else operation_value - commission

            # Determinar tipo de operación
            if value_diff > 0:
                operation_type = OperationType.BUY
                reason = f"Aumentar peso de {current_weight:.2%} a {target_weight:.2%}"
            else:
                operation_type = OperationType.SELL
                reason = f"Reducir peso de {current_weight:.2%} a {target_weight:.2%}"

            # Crear operación
            operation = RebalanceOperation(
                ticker=ticker,
                operation_type=operation_type,
                quantity=quantity,
                current_price=current_price,
                trade_value=operation_value,
                commission=commission,
                estimated_cost=estimated_cost,
                current_weight=current_weight,
                target_weight=target_weight,
                weight_diff=weight_diff,
                reason=reason
            )

            operations.append(operation)

            logger.info(
                f"{ticker}: {operation_type.value.upper()} {quantity:.4f} acciones "
                f"@ ${current_price:.2f} = ${operation_value:,.2f} "
                f"(peso: {current_weight:.2%} → {target_weight:.2%})"
            )

        # Ordenar operaciones: primero SELL, luego BUY (para liberar cash)
        operations.sort(key=lambda x: (x.operation_type != OperationType.SELL, abs(x.estimated_cost)), reverse=True)

        return operations

    def _generate_summary(
        self,
        operations: List[RebalanceOperation],
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_value: float
    ) -> Dict:
        """Genera un resumen del rebalanceo."""
        buy_operations = [op for op in operations if op.operation_type == OperationType.BUY]
        sell_operations = [op for op in operations if op.operation_type == OperationType.SELL]

        total_buy_value = sum(op.estimated_cost for op in buy_operations)
        total_sell_value = sum(op.estimated_cost for op in sell_operations)
        total_commission = sum(op.commission for op in operations)
        gross_trade_value = sum(op.trade_value for op in operations)

        # Estimaciones de costos adicionales (spread/impacto) simplificadas
        spread_cost = gross_trade_value * 0.0005
        market_impact = gross_trade_value * 0.0002
        total_cost = total_commission + spread_cost + market_impact

        # Calcular tracking error (desviación del objetivo)
        tracking_error_before = sum(
            abs(current_weights.get(ticker, 0) - target_weights.get(ticker, 0))
            for ticker in set(current_weights.keys()) | set(target_weights.keys())
        )

        # Estimar tracking error después (asumiendo ejecución perfecta)
        estimated_weights_after = current_weights.copy()
        for op in operations:
            if op.operation_type == OperationType.BUY:
                estimated_weights_after[op.ticker] = (
                    estimated_weights_after.get(op.ticker, 0) + (op.estimated_cost / total_value)
                )
            elif op.operation_type == OperationType.SELL:
                estimated_weights_after[op.ticker] = (
                    estimated_weights_after.get(op.ticker, 0) - (op.estimated_cost / total_value)
                )

        tracking_error_after = sum(
            abs(estimated_weights_after.get(ticker, 0) - target_weights.get(ticker, 0))
            for ticker in set(estimated_weights_after.keys()) | set(target_weights.keys())
        )

        return {
            'total_operations': len(operations),
            'buy_operations': len(buy_operations),
            'sell_operations': len(sell_operations),
            'total_buy_value': total_buy_value,
            'total_sell_value': total_sell_value,
            'net_cash_flow': total_sell_value - total_buy_value,
            'tracking_error_before': tracking_error_before,
            'tracking_error_after': tracking_error_after,
            'tracking_error_improvement': tracking_error_before - tracking_error_after,
            'costs': {
                'commission': total_commission,
                'spread_cost': spread_cost,
                'market_impact': market_impact,
                'total_cost': total_cost,
                'cost_percent': (total_cost / total_value) * 100 if total_value else 0
            }
        }

    def _format_weights(self, weights: Dict[str, float]) -> str:
        """Formatea pesos para logging."""
        if not weights:
            return "{}"

        items = [f"{ticker}: {weight:.2%}" for ticker, weight in sorted(weights.items())]
        return "{" + ", ".join(items) + "}"

    def validate_operations_feasibility(
        self,
        operations: List[RebalanceOperation],
        available_cash: float,
        current_positions: List[Position]
    ) -> Dict:
        """
        Valida que las operaciones sean factibles con el cash y posiciones disponibles.

        Args:
            operations: Lista de operaciones a validar
            available_cash: Cash disponible
            current_positions: Posiciones actuales

        Returns:
            Dict con:
                - feasible: bool
                - errors: List[str] - Errores encontrados
                - warnings: List[str] - Advertencias
        """
        errors = []
        warnings = []

        # Simular ejecución de operaciones
        simulated_cash = available_cash
        positions_map = {pos.ticker: pos.quantity for pos in current_positions}

        for op in operations:
            if op.operation_type == OperationType.BUY:
                # Verificar que haya suficiente cash
                if simulated_cash < op.estimated_cost:
                    errors.append(
                        f"Cash insuficiente para comprar {op.ticker}: "
                        f"${simulated_cash:.2f} disponible, ${op.estimated_cost:.2f} requerido"
                    )
                else:
                    simulated_cash -= op.estimated_cost

            elif op.operation_type == OperationType.SELL:
                # Verificar que haya suficientes acciones
                current_quantity = positions_map.get(op.ticker, 0)
                if current_quantity < op.quantity:
                    errors.append(
                        f"Acciones insuficientes de {op.ticker}: "
                        f"{current_quantity:.4f} disponibles, {op.quantity:.4f} requeridas"
                    )
                else:
                    simulated_cash += op.estimated_cost
                    positions_map[op.ticker] = current_quantity - op.quantity

        # Cash final
        if simulated_cash < 0:
            warnings.append(f"Cash final negativo: ${simulated_cash:.2f}")

        return {
            'feasible': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'simulated_final_cash': simulated_cash
        }
