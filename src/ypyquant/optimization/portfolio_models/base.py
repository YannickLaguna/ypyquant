"""
Contratos base de los modelos de portafolio: enums de tipos y clases abstractas (Strategy).

Responsabilidad: definir lo que comparten todos los modelos, sin implementar ningún algoritmo:
- `ReturnModelType`, `RiskModelType`, `OptimizerType`: identificadores de los tipos construibles.
- `ExpectedReturnsModel` (µ), `RiskModel` (Σ) y `PortfolioOptimizer` (pesos): las tres interfaces del patrón
  Strategy. `PortfolioOptimizer.calculate_metrics` y `validate_inputs` son comunes a todos los optimizadores.
- `_validar_retornos`: validación centralizada de la matriz de retornos que usan los modelos.

Se relaciona con: modelos_retorno.py, modelos_riesgo.py y optimizadores.py (las implementan),
modelo.py (los compone en `PortfolioOptimizationModel`) y con data_preprocessing.py (calidad de datos).
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _validar_retornos(df: pd.DataFrame) -> None:
    """
    Validación centralizada de DataFrames de retornos.

    I: DataFrame de retornos (columnas=tickers).
    T: Comprueba vacío, NaN e infinitos.
    O: None — lanza ValueError si alguna condición falla.
    """
    if df.empty:
        raise ValueError("Returns dataframe is empty")
    if df.isnull().any().any():
        raise ValueError("Returns dataframe contains NaN values")
    if np.isinf(df.values).any():
        columnas_infinitos = [col for col in df.columns if np.isinf(df[col]).any()]
        raise ValueError(
            f"Returns dataframe contiene valores infinitos en columnas: {columnas_infinitos}"
        )


class ReturnModelType(str, Enum):
    """Tipos de modelos para calcular rendimientos esperados"""
    # Históricos (v0.3.0): los usa crear_modelo_personalizado y la API de GrowSave
    # ("historical_mean"); antes faltaban y create_custom_model fallaba siempre.
    HISTORICAL_MEAN = "historical_mean"
    HISTORICAL_MEDIAN = "historical_median"
    CAPM = "capm"
    FAMA_FRENCH_4 = "fama_french_4"
    BLACK_LITTERMAN = "black_litterman"
    MACHINE_LEARNING = "machine_learning"
    CUSTOM = "custom"
    # Estimación robusta del TFM (v0.4.0)
    BAYES_STEIN = "bayes_stein"
    BL_PRIOR = "bl_prior"
    BL_MOMENTUM = "bl_momentum"


class RiskModelType(str, Enum):
    """Tipos de modelos para calcular matriz de riesgo"""
    SAMPLE_COVARIANCE = "sample_covariance"
    LEDOIT_WOLF = "ledoit_wolf"
    EXPONENTIAL_WEIGHTED = "exponential_weighted"
    SEMI_COVARIANCE = "semi_covariance"
    ORACLE_APPROXIMATING = "oracle_approximating"
    CUSTOM = "custom"


class OptimizerType(str, Enum):
    """Tipos de optimizadores"""
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    MAX_QUADRATIC_UTILITY = "max_quadratic_utility"
    EFFICIENT_RISK = "efficient_risk"
    EFFICIENT_RETURN = "efficient_return"
    RISK_PARITY = "risk_parity"
    HRP = "hrp"
    BLACK_LITTERMAN = "black_litterman"
    MIN_CVAR = "min_cvar"
    MAX_SORTINO = "max_sortino"
    # Estimación robusta del TFM (v0.4.0)
    MAX_SHARPE_QP = "max_sharpe_qp"
    MICHAUD = "michaud"
    # Optimización robusta con incertidumbre en µ (v0.5.0)
    ROBUSTO_ELIPSOIDE = "robusto_elipsoide"
    ROBUSTO_CAJA = "robusto_caja"


class ExpectedReturnsModel(ABC):
    """
    Base abstracta para estimadores del vector de retornos esperados (μ).

    I: DataFrame de retornos históricos (columnas=tickers).
    T: Cada subclase estima μ con su método particular (media, CAPM, BL…).
    O: pd.Series con retornos esperados anualizados por ticker.
    """

    def __init__(self, name: str = "base_returns"):
        self.name = name
        self.config = {}
        self.metadata = {}

    @abstractmethod
    def calculate(self, returns_df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calcula rendimientos esperados para cada activo.

        Args:
            returns_df: DataFrame con retornos históricos (columnas = tickers)
            **kwargs: Parámetros adicionales del modelo

        Returns:
            Series con rendimientos esperados anualizados por ticker
        """

    def validate_returns(self, returns_df: pd.DataFrame) -> None:
        """Validación común de datos — delega a _validar_retornos centralizado."""
        _validar_retornos(returns_df)

    def to_dict(self) -> dict:
        """Serializa configuración del modelo"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config,
            'metadata': self.metadata
        }


class RiskModel(ABC):
    """
    Base abstracta para estimadores de la matriz de covarianza (Σ).

    I: DataFrame de retornos históricos (columnas=tickers).
    T: Cada subclase estima Σ (muestral, LW, semi-cov…) y la anualiza.
    O: pd.DataFrame cuadrado con la matriz de covarianza anualizada.
    """

    def __init__(self, name: str = "base_risk"):
        self.name = name
        self.config = {}
        self.metadata = {}

    @abstractmethod
    def calculate(self, returns_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calcula matriz de covarianza/riesgo.

        Args:
            returns_df: DataFrame con retornos históricos
            **kwargs: Parámetros adicionales

        Returns:
            DataFrame con matriz de covarianza anualizada
        """

    def validate_returns(self, returns_df: pd.DataFrame) -> None:
        """Validación común — delega a _validar_retornos centralizado."""
        _validar_retornos(returns_df)

    def to_dict(self) -> dict:
        """Serializa configuración del modelo"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config,
            'metadata': self.metadata
        }


class PortfolioOptimizer(ABC):
    """
    Base abstracta para optimizadores de portafolio.

    I: μ (Series), Σ (DataFrame), restricciones (Dict opcional).
    T: Cada subclase resuelve un problema de optimización diferente.
    O: Dict con 'weights' (ticker→peso), métricas (retorno, vol, Sharpe) y metadata.
    """

    def __init__(self, name: str = "base_optimizer"):
        self.name = name
        self.config = {}
        self.metadata = {}

    @abstractmethod
    def optimize(
        self,
        expected_returns: pd.Series,
        risk_matrix: pd.DataFrame,
        constraints: dict | None = None,
        **kwargs
    ) -> dict:
        """
        Optimiza el portafolio y retorna pesos y métricas.

        Args:
            expected_returns: Series con rendimientos esperados
            risk_matrix: Matriz de covarianza
            constraints: Restricciones de optimización
            **kwargs: Parámetros adicionales

        Returns:
            Dict con weights, expected_return, volatility, sharpe_ratio, etc.
        """

    def validate_inputs(
        self,
        expected_returns: pd.Series,
        risk_matrix: pd.DataFrame
    ) -> None:
        """Validación común de entradas"""
        if expected_returns.empty:
            raise ValueError("Expected returns is empty")
        if risk_matrix.empty:
            raise ValueError("Risk matrix is empty")
        if not all(expected_returns.index == risk_matrix.index):
            raise ValueError("Tickers mismatch between returns and risk matrix")

    def calculate_metrics(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        risk_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> dict:
        """Calcula métricas estándar del portafolio"""
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(risk_matrix, weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        return {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_vol),
            'sharpe_ratio': float(sharpe)
        }

    def to_dict(self) -> dict:
        """Serializa configuración del optimizador"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config,
            'metadata': self.metadata
        }
