"""
RL Optimizer - Optimización de portafolios con Reinforcement Learning.
Framework esquemático extensible para RL.

Nota: Esta es una implementación esquemática. Para producción,
se recomienda usar Stable-Baselines3, Ray RLlib, o TensorTrade.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Action(Enum):
    """Acciones posibles del agente"""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class PortfolioState:
    """Estado del portafolio para RL"""
    cash: float
    positions: dict[str, float]  # {ticker: quantity}
    prices: dict[str, float]  # {ticker: current_price}
    features: dict[str, np.ndarray]  # Features adicionales

    @property
    def total_value(self) -> float:
        """Calcula valor total del portafolio."""
        positions_value = sum(
            qty * self.prices.get(ticker, 0)
            for ticker, qty in self.positions.items()
        )
        return self.cash + positions_value


class PortfolioEnvironment:
    """
    Environment de RL para optimización de portafolios.

    Compatible con la interfaz de Gym/Gymnasium.
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001
    ):
        """
        Inicializa el environment.

        Args:
            data: Dict con DataFrames de precios por ticker
            initial_capital: Capital inicial
            commission_rate: Tasa de comisión
        """
        self.data = data
        self.tickers = list(data.keys())
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate

        # Estado interno
        self.reset()

        logger.info(
            f"PortfolioEnvironment inicializado con {len(self.tickers)} tickers, "
            f"${initial_capital:,.2f}"
        )

    def reset(self) -> PortfolioState:
        """Resetea el environment."""
        self.current_step = 0
        self.cash = self.initial_capital
        self.positions = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_history = []

        return self._get_state()

    def step(self, actions: dict[str, Action]) -> tuple[PortfolioState, float, bool, dict]:
        """
        Ejecuta un paso en el environment.

        Args:
            actions: Dict de acciones por ticker

        Returns:
            Tuple (new_state, reward, done, info)
        """
        # Guardar portfolio value antes de la acción
        prev_value = self._get_portfolio_value()

        # Ejecutar acciones
        for ticker, action in actions.items():
            self._execute_action(ticker, action)

        # Avanzar un paso
        self.current_step += 1

        # Calcular nuevo portfolio value
        current_value = self._get_portfolio_value()

        # Calcular reward (retorno porcentual)
        reward = (current_value - prev_value) / prev_value if prev_value > 0 else 0

        # Guardar en historial
        self.portfolio_history.append({
            'step': self.current_step,
            'value': current_value,
            'cash': self.cash,
            'positions': self.positions.copy()
        })

        # Verificar si terminó
        done = self._is_done()

        # Info adicional
        info = {
            'portfolio_value': current_value,
            'cash': self.cash,
            'step': self.current_step
        }

        new_state = self._get_state()

        return new_state, reward, done, info

    def _execute_action(self, ticker: str, action: Action):
        """Ejecuta una acción."""
        current_price = self._get_current_price(ticker)

        if current_price is None:
            return

        if action == Action.BUY:
            # Comprar con 10% del cash disponible
            amount = self.cash * 0.1
            commission = amount * self.commission_rate
            shares = (amount - commission) / current_price

            if shares > 0:
                self.cash -= amount
                self.positions[ticker] += shares

        elif action == Action.SELL:
            # Vender 50% de la posición
            shares_to_sell = self.positions[ticker] * 0.5

            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price
                commission = proceeds * self.commission_rate
                self.cash += (proceeds - commission)
                self.positions[ticker] -= shares_to_sell

        # HOLD no hace nada

    def _get_current_price(self, ticker: str) -> float | None:
        """Obtiene precio actual de un ticker."""
        if ticker not in self.data:
            return None

        df = self.data[ticker]

        if self.current_step >= len(df):
            return None

        return float(df.iloc[self.current_step]['Close'])

    def _get_portfolio_value(self) -> float:
        """Calcula valor total del portafolio."""
        positions_value = 0.0

        for ticker, qty in self.positions.items():
            price = self._get_current_price(ticker)
            if price:
                positions_value += qty * price

        return self.cash + positions_value

    def _get_state(self) -> PortfolioState:
        """Obtiene estado actual."""
        prices = {}
        features = {}

        for ticker in self.tickers:
            price = self._get_current_price(ticker)
            if price:
                prices[ticker] = price

                # Features: últimos N precios normalizados
                df = self.data[ticker]
                lookback = 20
                start_idx = max(0, self.current_step - lookback)
                recent_prices = df.iloc[start_idx:self.current_step + 1]['Close'].values

                # Normalizar
                if len(recent_prices) > 0:
                    normalized = (recent_prices - recent_prices.mean()) / (recent_prices.std() + 1e-8)
                    features[ticker] = normalized
                else:
                    features[ticker] = np.array([0.0])

        return PortfolioState(
            cash=self.cash,
            positions=self.positions.copy(),
            prices=prices,
            features=features
        )

    def _is_done(self) -> bool:
        """Verifica si el episodio terminó."""
        # Termina si llegamos al final de los datos
        for df in self.data.values():
            if self.current_step >= len(df) - 1:
                return True

        # Termina si nos quedamos sin dinero
        if self._get_portfolio_value() < self.initial_capital * 0.1:
            return True

        return False


class RLOptimizer:
    """
    Optimizador de portafolios usando Reinforcement Learning.

    Framework esquemático para RL portfolio optimization.
    """

    def __init__(
        self,
        env: PortfolioEnvironment,
        algorithm: str = "random"  # 'random', 'dqn', 'a2c', 'ppo'
    ):
        """
        Inicializa el optimizador RL.

        Args:
            env: PortfolioEnvironment
            algorithm: Algoritmo de RL a usar
        """
        self.env = env
        self.algorithm = algorithm
        self.policy = None

        logger.info(f"RLOptimizer inicializado con algoritmo: {algorithm}")

    def train(self, episodes: int = 100) -> dict:
        """
        Entrena el agente RL.

        Args:
            episodes: Número de episodios de entrenamiento

        Returns:
            Dict con métricas de training
        """
        logger.info(f"Iniciando training por {episodes} episodios")

        episode_rewards = []
        final_values = []

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Seleccionar acción (random policy como baseline)
                actions = self._select_actions(state)

                # Ejecutar paso
                state, reward, done, info = self.env.step(actions)
                episode_reward += reward

            episode_rewards.append(episode_reward)
            final_values.append(info['portfolio_value'])

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_value = np.mean(final_values[-10:])
                logger.info(
                    f"Episode {episode + 1}/{episodes}: "
                    f"Avg Reward={avg_reward:.4f}, "
                    f"Avg Value=${avg_value:,.2f}"
                )

        metrics = {
            'total_episodes': episodes,
            'avg_reward': np.mean(episode_rewards),
            'avg_final_value': np.mean(final_values),
            'best_final_value': max(final_values),
            'worst_final_value': min(final_values)
        }

        logger.info(f"Training completado: Avg Final Value=${metrics['avg_final_value']:,.2f}")

        return metrics

    def _select_actions(self, state: PortfolioState) -> dict[str, Action]:
        """
        Selecciona acciones basado en el estado.

        Args:
            state: Estado actual del portafolio

        Returns:
            Dict de acciones por ticker
        """
        actions = {}

        for ticker in self.env.tickers:
            if self.algorithm == "random":
                # Random policy
                actions[ticker] = np.random.choice(list(Action))
            else:
                # TODO: Implementar políticas más sofisticadas (DQN, A2C, PPO)
                actions[ticker] = Action.HOLD

        return actions

    def optimize_weights(self, state: PortfolioState) -> dict[str, float]:
        """
        Optimiza pesos del portafolio usando RL.

        Args:
            state: Estado actual

        Returns:
            Dict con pesos óptimos por ticker
        """
        # Esta es una implementación simple
        # En producción, usarías el policy network entrenado

        total_value = state.total_value

        # Calcular valor actual de cada posición
        position_values = {}
        for ticker, qty in state.positions.items():
            price = state.prices.get(ticker, 0)
            position_values[ticker] = qty * price

        # Convertir a pesos
        weights = {}
        for ticker in self.env.tickers:
            value = position_values.get(ticker, 0)
            weights[ticker] = value / total_value if total_value > 0 else 0

        # Normalizar
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {t: w / total_weight for t, w in weights.items()}

        return weights


def create_simple_rl_strategy(
    tickers: list[str],
    historical_data: dict[str, pd.DataFrame],
    initial_capital: float = 10000.0
) -> tuple[RLOptimizer, dict]:
    """
    Crea y entrena una estrategia RL simple.

    Args:
        tickers: Lista de tickers
        historical_data: Datos históricos
        initial_capital: Capital inicial

    Returns:
        Tuple (optimizer, training_metrics)
    """
    # Crear environment
    env = PortfolioEnvironment(
        data=historical_data,
        initial_capital=initial_capital,
        commission_rate=0.001
    )

    # Crear optimizer
    optimizer = RLOptimizer(env, algorithm="random")

    # Entrenar
    metrics = optimizer.train(episodes=50)

    return optimizer, metrics
