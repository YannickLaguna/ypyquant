"""
Modelos de retorno, riesgo y optimización de portafolios (Strategy Pattern).

Clases principales:
- ExpectedReturnsModel / RiskModel / PortfolioOptimizer — bases abstractas.
- HistoricalReturnsModel, CAPMReturnsModel, BlackLittermanReturnsModel — estimadores de μ.
- SampleCovarianceModel, LedoitWolfShrinkageModel, SemiCovarianceModel — estimadores de Σ.
- MarkowitzOptimizer — max Sharpe / min vol via pypfopt.EfficientFrontier (fallback SLSQP).
- PMPTMinCVaROptimizer — min CVaR via pypfopt.EfficientCVaR (fallback SLSQP empírico).
- PMPTMaxSortinoOptimizer — max Sortino via pypfopt.EfficientSemivariance (fallback SLSQP).

Relacionado con: walk_forward.py (consume optimizadores), factor_models.py (provee μ),
black_litterman.py (posterior BL), data_preprocessing.py (valida retornos).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIÓN AUXILIAR CENTRALIZADA DE VALIDACIÓN
# ============================================================================

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


# ============================================================================
# ENUMS para configuración
# ============================================================================

class ReturnModelType(str, Enum):
    """Tipos de modelos para calcular rendimientos esperados"""
    CAPM = "capm"
    FAMA_FRENCH_4 = "fama_french_4"
    BLACK_LITTERMAN = "black_litterman"
    MACHINE_LEARNING = "machine_learning"
    CUSTOM = "custom"


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


# ============================================================================
# CLASES BASE ABSTRACTAS
# ============================================================================

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
        pass

    def validate_returns(self, returns_df: pd.DataFrame) -> None:
        """Validación común de datos — delega a _validar_retornos centralizado."""
        _validar_retornos(returns_df)

    def to_dict(self) -> Dict:
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
        pass

    def validate_returns(self, returns_df: pd.DataFrame) -> None:
        """Validación común — delega a _validar_retornos centralizado."""
        _validar_retornos(returns_df)

    def to_dict(self) -> Dict:
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
        constraints: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
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
        pass

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
    ) -> Dict:
        """Calcula métricas estándar del portafolio"""
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(risk_matrix, weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        return {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_vol),
            'sharpe_ratio': float(sharpe)
        }

    def to_dict(self) -> Dict:
        """Serializa configuración del optimizador"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config,
            'metadata': self.metadata
        }


# ============================================================================
# IMPLEMENTACIONES: MODELOS DE RENDIMIENTOS
# ============================================================================

class HistoricalReturnsModel(ExpectedReturnsModel):
    """Rendimientos históricos (media o mediana)"""

    def __init__(self, method: str = "mean", frequency: int = 252):
        super().__init__(name=f"historical_{method}")
        self.method = method
        self.frequency = frequency
        self.config = {'method': method, 'frequency': frequency}

    def calculate(self, returns_df: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_returns(returns_df)

        if self.method == "mean":
            returns = returns_df.mean() * self.frequency
        elif self.method == "median":
            returns = returns_df.median() * self.frequency
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.metadata['calculation_date'] = datetime.now().isoformat()
        return returns


class CAPMReturnsModel(ExpectedReturnsModel):
    """
    Capital Asset Pricing Model con tasas dinámicas de QuantLib.
    E[R] = Rf(plazo) + β(Rm - Rf)
    """

    def __init__(
        self,
        market_ticker: str = "^GSPC",
        risk_free_rate: float = 0.02,
        frequency: int = 252,
        usar_tasas_dinamicas: bool = True,
        horizonte_inversion_años: float = 5.0
    ):
        super().__init__(name="capm")
        self.market_ticker = market_ticker
        self.frequency = frequency
        self.usar_tasas_dinamicas = usar_tasas_dinamicas
        self.horizonte_inversion_años = horizonte_inversion_años

        # Configurar tasa libre de riesgo
        self._servicio_tasas = None
        if usar_tasas_dinamicas:
            try:
                from ypyquant.data.risk_free_rates import TasaLibreRiesgoDinamica
                self._servicio_tasas = TasaLibreRiesgoDinamica(actualizar_curva=True)
                self.risk_free_rate = self._servicio_tasas.obtener_tasa_por_plazo(horizonte_inversion_años)
                logger.debug(f"[CAPMReturns] Usando tasa dinámica {horizonte_inversion_años}Y: {self.risk_free_rate:.4f}")
            except Exception as e:
                logger.warning(f"[CAPMReturns] No se pudo usar QuantLib: {e}. Usando tasa fija.")
                self.risk_free_rate = risk_free_rate
                self.usar_tasas_dinamicas = False
        else:
            self.risk_free_rate = risk_free_rate

        self.config = {
            'market_ticker': market_ticker,
            'risk_free_rate': self.risk_free_rate,
            'frequency': frequency,
            'usar_tasas_dinamicas': self.usar_tasas_dinamicas,
            'horizonte_inversion_años': horizonte_inversion_años
        }

    def calculate(
        self,
        returns_df: pd.DataFrame,
        market_returns: Optional[pd.Series] = None,
        **kwargs
    ) -> pd.Series:
        self.validate_returns(returns_df)

        if market_returns is None:
            raise ValueError("CAPM requires market_returns parameter")

        # Alinear fechas
        aligned_data = returns_df.join(market_returns.rename('market'), how='inner')
        market_col = aligned_data['market']
        assets_df = aligned_data.drop('market', axis=1)

        # Calcular betas por ticker
        betas = {}
        for ticker in assets_df.columns:
            covariance = assets_df[ticker].cov(market_col)
            market_var = market_col.var()
            betas[ticker] = covariance / market_var if market_var > 0 else 1.0

        # Calcular market premium
        market_premium = market_col.mean() * self.frequency - self.risk_free_rate

        # E[R] = Rf(activo) + β * (E[Rm] - Rf)
        # Si usar_tasas_dinamicas, cada activo usa tasa apropiada
        expected_returns = {}
        for ticker in assets_df.columns:
            if self.usar_tasas_dinamicas and self._servicio_tasas is not None:
                try:
                    rf_activo = self._servicio_tasas.obtener_tasa_para_activo(ticker, betas[ticker])
                except Exception:
                    rf_activo = self.risk_free_rate
            else:
                rf_activo = self.risk_free_rate

            expected_returns[ticker] = rf_activo + betas[ticker] * market_premium

        self.metadata['betas'] = betas
        self.metadata['market_premium'] = float(market_premium)
        self.metadata['usar_tasas_dinamicas'] = self.usar_tasas_dinamicas
        return pd.Series(expected_returns)


class BlackLittermanReturnsModel(ExpectedReturnsModel):
    """Black-Litterman Model: Combina equilibrio de mercado con views del inversor"""

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.025,
        market_caps: Optional[Dict[str, float]] = None
    ):
        super().__init__(name="black_litterman")
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.market_caps = market_caps or {}
        self.config = {
            'risk_aversion': risk_aversion,
            'tau': tau,
            'has_market_caps': bool(market_caps)
        }

    def calculate(
        self,
        returns_df: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None,
        views: Optional[Dict[str, float]] = None,
        view_confidences: Optional[List[float]] = None,
        **kwargs
    ) -> pd.Series:
        self.validate_returns(returns_df)

        # Verificar datos de entrada
        logger.debug(f"[BlackLitterman] returns_df shape: {returns_df.shape}, "
                     f"NaN={returns_df.isna().any().any()}, Inf={np.isinf(returns_df.values).any()}")
        if np.isinf(returns_df.values).any():
            cols_inf = [c for c in returns_df.columns if np.isinf(returns_df[c]).any()]
            logger.warning(f"[BlackLitterman] Infinitos detectados en returns_df: {cols_inf}")

        try:
            from pypfopt import black_litterman, risk_models

            # Calcular covarianza si no se proporciona
            if cov_matrix is None:
                logger.debug("[BlackLitterman] Calculando covarianza con risk_models.sample_cov...")
                cov_matrix = risk_models.sample_cov(
                    returns_df,
                    returns_data=True,  # Ya son retornos, no precios
                    frequency=252
                )
                logger.debug(f"[BlackLitterman] Covarianza calculada: shape={cov_matrix.shape}, "
                             f"NaN={cov_matrix.isna().any().any()}, Inf={np.isinf(cov_matrix.values).any()}")
                if np.isinf(cov_matrix.values).any():
                    logger.warning("[BlackLitterman] Infinitos en matriz de covarianza")

            # Prior: equilibrio de mercado
            if self.market_caps:
                market_weights = pd.Series(self.market_caps)
                market_weights = market_weights / market_weights.sum()
                prior = black_litterman.market_implied_prior_returns(
                    market_weights,
                    self.risk_aversion,
                    cov_matrix
                )
            else:
                # Sin market caps, usar retornos históricos como prior
                prior = returns_df.mean() * 252

            # Aplicar views si existen
            if views and view_confidences:
                try:
                    from ypyquant.optimization.black_litterman import BlackLittermanQuantLib, crear_views_desde_dict

                    logger.debug(f"[BlackLitterman] Usando QuantLib con {len(views)} views")

                    # Convertir views del formato antiguo al nuevo
                    views_list = []
                    for ticker, retorno_esperado in views.items():
                        views_list.append({
                            'tipo': 'absolute',
                            'activos': [ticker],
                            'valor_esperado': retorno_esperado,
                            'confianza': view_confidences[0] if view_confidences else 0.5
                        })

                    views_objects = crear_views_desde_dict(views_list)

                    # Crear modelo Black-Litterman
                    bl = BlackLittermanQuantLib(
                        risk_aversion=self.risk_aversion,
                        tau=self.tau,
                        metodo_omega='idzorek'
                    )

                    # Calcular posterior
                    if self.market_caps:
                        market_weights = pd.Series(self.market_caps)
                        market_weights = market_weights / market_weights.sum()
                    else:
                        # Sin market caps, pesos iguales
                        market_weights = pd.Series(1.0, index=cov_matrix.index)
                        market_weights = market_weights / market_weights.sum()

                    posterior = bl.calcular_posterior(
                        cov_matrix=cov_matrix,
                        market_weights=market_weights,
                        views=views_objects,
                        prior=prior
                    )

                    logger.debug("[BlackLitterman] Posterior calculado con QuantLib")

                except ImportError:
                    logger.warning("[BlackLitterman] QuantLib BL no disponible, usando prior sin ajustar")
                    posterior = prior
                except Exception as e:
                    logger.warning(f"[BlackLitterman] Error en BL QuantLib: {e}, usando prior")
                    posterior = prior
            else:
                posterior = prior

            self.metadata['has_views'] = bool(views)
            return posterior

        except ImportError:
            logger.warning("[BlackLitterman] PyPortfolioOpt no disponible, usando media histórica.")
            return returns_df.mean() * 252


# ============================================================================
# IMPLEMENTACIONES: MODELOS DE RIESGO
# ============================================================================

class SampleCovarianceModel(RiskModel):
    """Matriz de covarianza de muestra estándar"""

    def __init__(self, frequency: int = 252):
        super().__init__(name="sample_covariance")
        self.frequency = frequency
        self.config = {'frequency': frequency}

    def calculate(self, returns_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_returns(returns_df)
        cov_matrix = returns_df.cov() * self.frequency
        self.metadata['condition_number'] = float(np.linalg.cond(cov_matrix))
        return cov_matrix


class LedoitWolfShrinkageModel(RiskModel):
    """
    Estimador Ledoit-Wolf con shrinkage óptimo analítico.

    I: DataFrame de retornos diarios.
    T: pypfopt.CovarianceShrinkage.ledoit_wolf() (fallback: sklearn → muestral).
    O: Σ anualizada con menor error de estimación que la covarianza muestral.
    """

    def __init__(self, frequency: int = 252):
        super().__init__(name="ledoit_wolf")
        self.frequency = frequency
        self.config = {'frequency': frequency}

    def calculate(self, returns_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_returns(returns_df)

        logger.debug(f"[LedoitWolf] shape={returns_df.shape}, "
                     f"NaN={returns_df.isna().any().any()}, Inf={np.isinf(returns_df.values).any()}")
        if np.isinf(returns_df.values).any():
            cols_inf = [c for c in returns_df.columns if np.isinf(returns_df[c]).any()]
            logger.warning(f"[LedoitWolf] Infinitos detectados: {cols_inf}")

        try:
            from pypfopt import risk_models
            # IMPORTANTE: returns_data=True porque returns_df ya son retornos, no precios
            cov_matrix = risk_models.CovarianceShrinkage(
                returns_df,
                returns_data=True,  # ¡CRÍTICO! Evita calcular "retornos de retornos"
                frequency=self.frequency
            ).ledoit_wolf()
            self.metadata['method'] = 'pypfopt'
            logger.debug("[LedoitWolf] Completado con pypfopt")
            return cov_matrix
        except ImportError:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                lw.fit(returns_df)
                cov_matrix = pd.DataFrame(
                    lw.covariance_ * self.frequency,
                    index=returns_df.columns,
                    columns=returns_df.columns
                )
                self.metadata['method'] = 'sklearn'
                self.metadata['shrinkage'] = float(lw.shrinkage_)
                return cov_matrix
            except ImportError:
                logger.warning("[LedoitWolf] Fallback a covarianza muestral")
                self.metadata['method'] = 'fallback'
                return returns_df.cov() * self.frequency


class ExponentialCovarianceModel(RiskModel):
    """Exponentially Weighted Covariance: da más peso a observaciones recientes"""

    def __init__(self, span: int = 180, frequency: int = 252):
        super().__init__(name="exponential_weighted")
        self.span = span
        self.frequency = frequency
        self.config = {'span': span, 'frequency': frequency}

    def calculate(self, returns_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_returns(returns_df)

        try:
            from pypfopt import risk_models
            cov_matrix = risk_models.exp_cov(
                returns_df,
                returns_data=True,  # Ya son retornos, no precios
                span=self.span,
                frequency=self.frequency
            )
            self.metadata['method'] = 'pypfopt'
            return cov_matrix
        except ImportError:
            # Fallback manual con pandas
            ewm_returns = returns_df.ewm(span=self.span).mean()
            cov_matrix = ewm_returns.cov() * self.frequency
            self.metadata['method'] = 'pandas_ewm'
            return cov_matrix


class SemiCovarianceModel(RiskModel):
    """
    Matriz de semi-covarianza muestral (solo retornos por debajo del umbral τ).

    I: DataFrame de retornos; target_return=τ (típicamente 0 o rf diario).
    T: pypfopt.risk_models.semicovariance() (fallback: zeroing retornos > τ y cov()).
    O: Σ^down anualizada — usada por PMPTMaxSortinoOptimizer y PMPTMinSemivarOptimizer.
    """

    def __init__(self, target_return: float = 0.0, frequency: int = 252):
        super().__init__(name="semi_covariance")
        self.target_return = target_return
        self.frequency = frequency
        self.config = {'target_return': target_return, 'frequency': frequency}

    def calculate(self, returns_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_returns(returns_df)

        try:
            from pypfopt import risk_models as _rm
            semi_cov = _rm.semicovariance(
                returns_df,
                returns_data=True,
                benchmark=self.target_return,
                frequency=self.frequency,
            )
            self.metadata['method'] = 'pypfopt'
        except (ImportError, Exception) as exc:
            logger.warning(f"[SemiCovarianceModel] pypfopt no disponible ({exc}), usando cálculo manual")
            # Fallback: zeroing retornos por encima del umbral
            downside_returns = returns_df.copy()
            downside_returns[downside_returns > self.target_return] = 0
            semi_cov = downside_returns.cov() * self.frequency
            self.metadata['method'] = 'manual'

        self.metadata['downside_observations'] = int(
            (returns_df < self.target_return).sum().sum()
        )
        return semi_cov


class LedoitWolfSemiCovarianceModel(RiskModel):
    """
    Semi-covarianza con shrinkage Ledoit-Wolf.

    Aplica el estimador óptimo de shrinkage (OAS) de Ledoit-Wolf sobre los
    retornos downside (filtrados por debajo del umbral τ). Esto permite que
    la elección de Σ (muestra vs LW) tenga efecto real en PMPT_MIN_SEMIVAR,
    diferenciando los resultados entre ambos métodos de estimación.

    Pipeline:
    1. Filtrar retornos: poner a 0 los que están por encima de τ (=rf diario)
    2. Aplicar shrinkage LW con sklearn.covariance.LedoitWolf sobre retornos downside
    3. Anualizar con * frequency

    Se relaciona con:
    - SemiCovarianceModel: versión sin shrinkage (covarianza muestral downside)
    - walk_forward.py: usado cuando metodo_sigma=LEDOIT_WOLF y riesgo=PMPT_MIN_SEMIVAR
    """

    def __init__(self, target_return: float = 0.0, frequency: int = 252):
        """
        Args:
            target_return: Umbral τ (típicamente rf diario). Retornos por encima se anulan.
            frequency: Factor de anualización (252 para datos diarios)
        """
        super().__init__(name="ledoit_wolf_semi_covariance")
        self.target_return = target_return
        self.frequency = frequency
        self.config = {'target_return': target_return, 'frequency': frequency}

    def calculate(self, returns_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_returns(returns_df)
        from sklearn.covariance import LedoitWolf

        # Solo retornos por debajo del umbral τ (downside)
        downside_returns = returns_df.copy()
        downside_returns[downside_returns > self.target_return] = 0

        # Aplicar shrinkage LW sobre los retornos downside
        lw = LedoitWolf()
        lw.fit(downside_returns)

        semi_cov = pd.DataFrame(
            lw.covariance_ * self.frequency,
            index=returns_df.columns,
            columns=returns_df.columns
        )

        self.metadata['shrinkage'] = float(lw.shrinkage_)
        self.metadata['downside_observations'] = int((downside_returns != 0).sum().sum())
        return semi_cov


# ============================================================================
# IMPLEMENTACIONES: OPTIMIZADORES
# ============================================================================

class MarkowitzOptimizer(PortfolioOptimizer):
    """
    Optimización media-varianza clásica (Markowitz).

    I: μ (Series), Σ (DataFrame), objective='max_sharpe'|'min_volatility'.
    T: pypfopt.EfficientFrontier con backend CVXPY (fallback SLSQP si pypfopt falla).
    O: Pesos óptimos + métricas (retorno esperado, volatilidad, Sharpe).
    """

    def __init__(
        self,
        objective: str = "max_sharpe",
        risk_free_rate: float = 0.02
    ):
        super().__init__(name=f"markowitz_{objective}")
        self.objective = objective
        self.risk_free_rate = risk_free_rate
        self.config = {'objective': objective, 'risk_free_rate': risk_free_rate}

    def optimize(
        self,
        expected_returns: pd.Series,
        risk_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        self.validate_inputs(expected_returns, risk_matrix)

        max_weight = constraints.get('max_weight', 1.0) if constraints else 1.0

        # Intentar con pypfopt.EfficientFrontier (backend CVXPY, más estable numéricamente)
        try:
            from pypfopt import EfficientFrontier as _EF
            from pypfopt.exceptions import OptimizationError as _OE

            ef = _EF(
                expected_returns,
                risk_matrix,
                weight_bounds=(0, max_weight),
            )
            if self.objective == "max_sharpe":
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            elif self.objective == "min_volatility":
                ef.min_volatility()
            else:
                raise ValueError(f"Unknown objective: {self.objective}")

            weights_dict = ef.clean_weights()
            weights = np.array([weights_dict.get(t, 0.0) for t in expected_returns.index])
            metrics = self.calculate_metrics(weights, expected_returns, risk_matrix, self.risk_free_rate)
            return {
                'weights': {t: float(w) for t, w in zip(expected_returns.index, weights)},
                **metrics,
                'optimizer_metadata': {'method': 'pypfopt_ef', 'success': True},
            }

        except (ImportError, Exception) as exc:
            logger.warning(f"[MarkowitzOptimizer] pypfopt falló ({exc}), usando SLSQP")

        # Fallback: SLSQP manual
        from scipy.optimize import minimize

        num_assets = len(expected_returns)
        initial_weights = np.array([1 / num_assets] * num_assets)

        if self.objective == "max_sharpe":
            def objective_func(w):
                ret = np.sum(expected_returns * w)
                vol = np.sqrt(np.dot(w.T, np.dot(risk_matrix, w)))
                sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
                return -sharpe
        elif self.objective == "min_volatility":
            def objective_func(w):
                return np.sqrt(np.dot(w.T, np.dot(risk_matrix, w)))
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, max_weight) for _ in range(num_assets))

        result = minimize(
            objective_func,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000},
        )

        if not result.success:
            logger.warning(f"[MarkowitzOptimizer] SLSQP no convergió: {result.message}")

        weights = result.x
        metrics = self.calculate_metrics(weights, expected_returns, risk_matrix, self.risk_free_rate)
        return {
            'weights': {ticker: float(w) for ticker, w in zip(expected_returns.index, weights)},
            **metrics,
            'optimizer_metadata': {
                'method': 'slsqp',
                'success': result.success,
                'message': result.message,
                'iterations': result.nit,
            },
        }


class RiskParityOptimizer(PortfolioOptimizer):
    """Risk Parity: cada activo contribuye igual al riesgo total"""

    def __init__(self):
        super().__init__(name="risk_parity")

    def optimize(
        self,
        expected_returns: pd.Series,
        risk_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        self.validate_inputs(expected_returns, risk_matrix)

        from scipy.optimize import minimize

        num_assets = len(expected_returns)

        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(risk_matrix, weights)))
            if portfolio_vol == 0:
                return 1e10
            marginal_contrib = np.dot(risk_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            target_risk = portfolio_vol / num_assets
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        max_weight = constraints.get('max_weight', 1.0) if constraints else 1.0
        bounds = tuple((0, max_weight) for _ in range(num_assets))
        initial_weights = np.array([1/num_assets] * num_assets)

        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )

        weights = result.x
        metrics = self.calculate_metrics(weights, expected_returns, risk_matrix)

        return {
            'weights': {ticker: float(w) for ticker, w in zip(expected_returns.index, weights)},
            **metrics,
            'optimizer_metadata': {
                'success': result.success,
                'message': result.message
            }
        }


# ============================================================================
# CLASE PRINCIPAL: PORTFOLIO OPTIMIZATION MODEL
# ============================================================================

class PortfolioOptimizationModel:
    """
    Clase principal para configurar y ejecutar optimización modular.
    Permite composición de diferentes componentes.

    TODO: Revisar lógica, no todo es compatible con todo
    """

    def __init__(
        self,
        name: str,
        returns_model: ExpectedReturnsModel,
        risk_model: RiskModel,
        optimizer: PortfolioOptimizer,
        description: str = ""
    ):
        self.name = name
        self.returns_model = returns_model
        self.risk_model = risk_model
        self.optimizer = optimizer
        self.description = description
        self.created_at = datetime.now()
        self.metadata = {}

    def optimize(
        self,
        returns_df: pd.DataFrame,
        constraints: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Ejecuta optimización completa: calcula returns, risk y optimiza.

        Args:
            returns_df: DataFrame con retornos históricos
            constraints: Restricciones de optimización
            **kwargs: Parámetros adicionales para los modelos

        Returns:
            Dict con resultados completos de optimización
        """
        # 1. Calcular rendimientos esperados
        expected_returns = self.returns_model.calculate(returns_df, **kwargs)

        # 2. Calcular matriz de riesgo
        risk_matrix = self.risk_model.calculate(returns_df, **kwargs)

        # 3. Optimizar
        result = self.optimizer.optimize(
            expected_returns,
            risk_matrix,
            constraints,
            **kwargs
        )

        # Añadir metadata del modelo
        result['model_info'] = {
            'name': self.name,
            'description': self.description,
            'returns_model': self.returns_model.to_dict(),
            'risk_model': self.risk_model.to_dict(),
            'optimizer': self.optimizer.to_dict(),
            'created_at': self.created_at.isoformat(),
            'optimized_at': datetime.now().isoformat()
        }

        return result

    def to_dict(self) -> Dict:
        """Serializa configuración completa del modelo"""
        return {
            'name': self.name,
            'description': self.description,
            'returns_model': self.returns_model.to_dict(),
            'risk_model': self.risk_model.to_dict(),
            'optimizer': self.optimizer.to_dict(),
            'created_at': self.created_at.isoformat()
        }

    def __repr__(self):
        return (
            f"PortfolioOptimizationModel("
            f"name='{self.name}', "
            f"returns={self.returns_model.name}, "
            f"risk={self.risk_model.name}, "
            f"optimizer={self.optimizer.name})"
        )


# ============================================================================
# FACTORY FUNCTIONS para crear modelos preconfigurados
# ============================================================================

def create_classic_markowitz_model() -> PortfolioOptimizationModel:
    """Modelo clásico de Markowitz: CAPM Returns + Ledoit-Wolf Shrinkage + Max Sharpe"""
    return PortfolioOptimizationModel(
        name="Classic Markowitz",
        returns_model=CAPMReturnsModel(),
        risk_model=LedoitWolfShrinkageModel(),
        optimizer=MarkowitzOptimizer(objective="max_sharpe"),
        description="Modelo clásico de Markowitz con CAPM y shrinkage óptimo de Ledoit-Wolf"
    )


def create_robust_model() -> PortfolioOptimizationModel:
    """Modelo robusto: Median Returns + Ledoit-Wolf + Risk Parity"""
    return PortfolioOptimizationModel(
        name="Robust Risk Parity",
        returns_model=HistoricalReturnsModel(method="median"),
        risk_model=LedoitWolfShrinkageModel(),
        optimizer=RiskParityOptimizer(),
        description="Modelo robusto con shrinkage y risk parity"
    )


def create_capm_model() -> PortfolioOptimizationModel:
    """Modelo CAPM: CAPM Returns + Ledoit-Wolf Shrinkage + Max Sharpe"""
    return PortfolioOptimizationModel(
        name="CAPM Max Sharpe",
        returns_model=CAPMReturnsModel(),
        risk_model=LedoitWolfShrinkageModel(),
        optimizer=MarkowitzOptimizer(objective="max_sharpe"),
        description="Modelo basado en CAPM con shrinkage óptimo de Ledoit-Wolf"
    )


# ============================================================================
# NUEVOS MODELOS CON MACHINE LEARNING
# ============================================================================

class PCADenoisedRiskModel(RiskModel):
    """Modelo de riesgo con denoising usando PCA"""

    def __init__(self, n_components: float = 0.95, frequency: int = 252):
        super().__init__(name="pca_denoised")
        self.n_components = n_components
        self.frequency = frequency
        self.config = {'n_components': n_components, 'frequency': frequency}

    def calculate(self, returns_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_returns(returns_df)

        try:
            from ypyquant.ml.pca_risk import PCARiskService

            pca_risk = PCARiskService(n_components=self.n_components)
            cov_matrix = pca_risk.calculate_denoised_covariance(returns_df, self.frequency)

            self.metadata['n_components_actual'] = pca_risk.pca.n_components_
            self.metadata['explained_variance'] = float(pca_risk.explained_variance_ratio.sum())
            self.metadata['method'] = 'pca_denoising'

            return cov_matrix

        except ImportError as e:
            logger.warning(f"[PCADenoisedRisk] PCA no disponible ({e}), usando Ledoit-Wolf")
            # Fallback a Ledoit-Wolf
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(returns_df)
            cov_matrix = pd.DataFrame(
                lw.covariance_ * self.frequency,
                index=returns_df.columns,
                columns=returns_df.columns
            )
            self.metadata['method'] = 'ledoit_wolf_fallback'
            return cov_matrix


class LSTMReturnsModel(ExpectedReturnsModel):
    """Modelo de retornos esperados usando LSTM"""

    def __init__(
        self,
        sequence_length: int = 60,
        forecast_horizon: int = 5,
        lstm_units: int = 50,
        dropout: float = 0.2,
        pretrained_models_path: Optional[str] = None
    ):
        super().__init__(name="lstm_predictor")
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.pretrained_models_path = pretrained_models_path
        self.config = {
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon,
            'lstm_units': lstm_units,
            'dropout': dropout
        }

    def calculate(self, returns_df: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_returns(returns_df)

        try:
            from ypyquant.ml.lstm_predictor import LSTMReturnsPredictor

            predictor = LSTMReturnsPredictor(
                sequence_length=self.sequence_length,
                forecast_horizon=self.forecast_horizon,
                lstm_units=self.lstm_units,
                dropout=self.dropout
            )

            # Si hay modelos pre-entrenados, cargarlos
            if self.pretrained_models_path:
                try:
                    predictor.load_models(self.pretrained_models_path)
                    logger.info(f"[LSTM] Usando modelos pre-entrenados desde {self.pretrained_models_path}")
                except Exception as e:
                    logger.warning(f"[LSTM] No se pudieron cargar modelos pre-entrenados: {e}. Entrenando desde cero...")
                    predictor.train(returns_df, epochs=30, validation_split=0.2, verbose=0)
            else:
                # Entrenar on-the-fly (puede ser lento)
                logger.info("[LSTM] Entrenando modelos LSTM desde cero...")
                predictor.train(returns_df, epochs=30, validation_split=0.2, verbose=0)

            # Predecir
            expected_returns = predictor.predict_expected_returns(returns_df, annualize=True)

            self.metadata['models_trained'] = len(predictor.models)
            self.metadata['tickers'] = list(predictor.models.keys())
            self.metadata['calculation_date'] = datetime.now().isoformat()

            return expected_returns

        except ImportError:
            logger.warning("[LSTM] TensorFlow no disponible, usando media histórica como fallback")
            self.metadata['method'] = 'historical_mean_fallback'
            return returns_df.mean() * 252


# ============================================================================
# OPTIMIZADORES PMPT (Post-Modern Portfolio Theory)
# ============================================================================

class PMPTMinCVaROptimizer(PortfolioOptimizer):
    """
    Minimiza el CVaR (Expected Shortfall) empírico del portafolio.

    I: μ, Σ, retornos_historicos (DataFrame, kwarg obligatorio), nivel_confianza.
    T: pypfopt.EfficientCVaR.min_cvar() (fallback SLSQP sobre CVaR empírico).
    O: Pesos + métricas + 'cvar' (pérdida esperada en el peor α% de días).
    """

    def __init__(self, nivel_confianza: float = 0.95, risk_free_rate: float = 0.02):
        """
        Args:
            nivel_confianza: Nivel de confianza para el CVaR (ej: 0.95 → peor 5%)
            risk_free_rate: Tasa libre de riesgo para cálculo de métricas
        """
        super().__init__(name="pmpt_min_cvar")
        self.nivel_confianza = nivel_confianza
        self.risk_free_rate = risk_free_rate
        self.config = {
            'nivel_confianza': nivel_confianza,
            'risk_free_rate': risk_free_rate
        }

    def _calcular_cvar(self, pesos: np.ndarray, retornos_hist: pd.DataFrame) -> float:
        """
        CVaR empírico del portafolio.

        I: pesos (array), retornos_hist (DataFrame diarios).
        T: Retornos del portafolio → percentil (1-α)% → media de la cola inferior.
        O: float — CVaR (negativo = pérdida).
        """
        retornos_portafolio = (retornos_hist * pesos).sum(axis=1)
        percentil = (1 - self.nivel_confianza) * 100
        var = np.percentile(retornos_portafolio, percentil)
        cola = retornos_portafolio[retornos_portafolio <= var]
        if len(cola) == 0:
            return var
        return float(cola.mean())

    def optimize(
        self,
        expected_returns: pd.Series,
        risk_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Minimiza el CVaR empírico del portafolio.

        Args:
            expected_returns: Series con rendimientos esperados (puede ser dummy si γ=∞)
            risk_matrix: Matriz de covarianza (usada para métricas, no para el objetivo)
            constraints: Dict con restricciones (ej: {'max_weight': 0.10})
            **kwargs:
                retornos_historicos (pd.DataFrame): REQUERIDO. Retornos diarios históricos.

        Returns:
            Dict con weights, métricas y metadata del optimizador
        """
        retornos_hist = kwargs.get('retornos_historicos')
        if retornos_hist is None:
            raise ValueError(
                "PMPTMinCVaROptimizer requiere 'retornos_historicos' como kwarg."
            )

        self.validate_inputs(expected_returns, risk_matrix)
        peso_maximo = constraints.get('max_weight', 1.0) if constraints else 1.0

        # Intentar con pypfopt.EfficientCVaR (backend CVXPY, más estable numéricamente)
        try:
            from pypfopt.efficient_frontier import EfficientCVaR as _ECVaR

            ec = _ECVaR(
                expected_returns,
                retornos_hist,
                beta=self.nivel_confianza,
                weight_bounds=(0, peso_maximo),
            )
            ec.min_cvar()
            weights_dict = ec.clean_weights()
            pesos = np.array([weights_dict.get(t, 0.0) for t in expected_returns.index])
            metricas = self.calculate_metrics(pesos, expected_returns, risk_matrix, self.risk_free_rate)
            cvar_final = self._calcular_cvar(pesos, retornos_hist)
            return {
                'weights': {t: float(w) for t, w in zip(expected_returns.index, pesos)},
                **metricas,
                'cvar': float(cvar_final),
                'nivel_confianza': self.nivel_confianza,
                'optimizer_metadata': {'method': 'pypfopt_ecvar', 'success': True},
            }

        except (ImportError, Exception) as exc:
            logger.warning(f"[PMPTMinCVaR] pypfopt falló ({exc}), usando SLSQP")

        # Fallback: SLSQP manual sobre CVaR empírico
        from scipy.optimize import minimize

        num_activos = len(expected_returns)
        pesos_iniciales = np.array([1 / num_activos] * num_activos)

        def objetivo(w):
            return -self._calcular_cvar(w, retornos_hist)

        restricciones = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, peso_maximo) for _ in range(num_activos))

        resultado = minimize(
            objetivo,
            pesos_iniciales,
            method='SLSQP',
            bounds=bounds,
            constraints=restricciones,
            options={'maxiter': 1000},
        )

        pesos = resultado.x
        metricas = self.calculate_metrics(pesos, expected_returns, risk_matrix, self.risk_free_rate)
        cvar_final = self._calcular_cvar(pesos, retornos_hist)
        return {
            'weights': {ticker: float(w) for ticker, w in zip(expected_returns.index, pesos)},
            **metricas,
            'cvar': float(cvar_final),
            'nivel_confianza': self.nivel_confianza,
            'optimizer_metadata': {
                'method': 'slsqp',
                'success': resultado.success,
                'message': resultado.message,
                'iterations': resultado.nit,
            },
        }


class PMPTMaxSortinoOptimizer(PortfolioOptimizer):
    """
    Maximiza el Sortino ratio: (μ'w - rf) / sqrt(w' Σ^down w).

    I: μ, Σ^down (semi-covarianza), retornos_historicos (kwarg opcional para pypfopt).
    T: pypfopt.EfficientSemivariance.min_semivariance() si hay retornos históricos
       (fallback SLSQP con función objetivo Sortino matricial).
    O: Pesos + 'sortino_ratio' + métricas estándar calculadas con Σ^down.
    """

    def __init__(self, risk_free_rate: float = 0.02, target_return: float = 0.0):
        """
        Args:
            risk_free_rate: Tasa libre de riesgo anualizada (τ en el numerador)
            target_return: MAR para el cálculo de Σ^down (no usado directamente aquí,
                           ya que la semi-covarianza se calcula externamente)
        """
        super().__init__(name="pmpt_max_sortino")
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.config = {
            'risk_free_rate': risk_free_rate,
            'target_return': target_return
        }

    def optimize(
        self,
        expected_returns: pd.Series,
        risk_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Maximiza el Sortino ratio: max (μ'w - rf) / sqrt(w' Σ^down w).

        La risk_matrix debe ser la semi-covarianza (Σ^down), no la covarianza
        completa. Esto se asegura desde walk_forward._optimizar().

        Args:
            expected_returns: Series con rendimientos esperados (μ anualizado)
            risk_matrix: Matriz de semi-covarianza anualizada (Σ^down)
            constraints: Dict con restricciones (ej: {'max_weight': 0.10})

        Returns:
            Dict con weights, métricas y metadata del optimizador
        """
        self.validate_inputs(expected_returns, risk_matrix)
        peso_maximo = constraints.get('max_weight', 1.0) if constraints else 1.0
        retornos_hist = kwargs.get('retornos_historicos')

        # Intentar con pypfopt.EfficientSemivariance cuando se dispongan retornos históricos
        # (pypfopt no tiene max_sortino nativo → usamos min_semivariance como aproximación CVXPY)
        if retornos_hist is not None:
            try:
                from pypfopt.efficient_frontier import EfficientSemivariance as _ESV

                esv = _ESV(
                    expected_returns,
                    retornos_hist,
                    frequency=252,
                    weight_bounds=(0, peso_maximo),
                )
                esv.min_semivariance()
                weights_dict = esv.clean_weights()
                pesos = np.array([weights_dict.get(t, 0.0) for t in expected_returns.index])
                mu_vals = expected_returns.values
                sigma_down_vals = risk_matrix.values
                retorno_p = float(np.dot(mu_vals, pesos))
                dv = float(np.sqrt(np.dot(pesos.T, np.dot(sigma_down_vals, pesos))))
                sortino_f = (retorno_p - self.risk_free_rate) / dv if dv > 1e-12 else 0.0
                metricas = {
                    'expected_return': retorno_p,
                    'volatility': dv,
                    'sharpe_ratio': sortino_f,
                }
                return {
                    'weights': {t: float(w) for t, w in zip(expected_returns.index, pesos)},
                    **metricas,
                    'sortino_ratio': float(sortino_f),
                    'target_return': self.target_return,
                    'optimizer_metadata': {'method': 'pypfopt_esv', 'success': True},
                }
            except (ImportError, Exception) as exc:
                logger.warning(f"[PMPTMaxSortino] pypfopt EfficientSemivariance falló ({exc}), usando SLSQP")

        # Fallback (y caso sin retornos históricos): SLSQP sobre Sortino matricial
        from scipy.optimize import minimize

        num_activos = len(expected_returns)
        pesos_iniciales = np.array([1 / num_activos] * num_activos)
        mu_vals = expected_returns.values
        sigma_down_vals = risk_matrix.values

        # Función objetivo matricial: -Sortino = -(μ'w - rf) / sqrt(w' Σ^down w)
        def objetivo(w):
            retorno = np.dot(mu_vals, w)
            downside_vol = np.sqrt(np.dot(w.T, np.dot(sigma_down_vals, w)))
            if downside_vol < 1e-12:
                return 0.0
            return -(retorno - self.risk_free_rate) / downside_vol

        restricciones = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, peso_maximo) for _ in range(num_activos))

        resultado = minimize(
            objetivo,
            pesos_iniciales,
            method='SLSQP',
            bounds=bounds,
            constraints=restricciones,
            options={'maxiter': 1000},
        )

        if not resultado.success:
            logger.warning(f"[PMPTMaxSortino] SLSQP no convergió: {resultado.message}")

        pesos = resultado.x
        retorno_portafolio = float(np.dot(mu_vals, pesos))
        downside_vol = float(np.sqrt(np.dot(pesos.T, np.dot(sigma_down_vals, pesos))))
        sortino_final = (
            (retorno_portafolio - self.risk_free_rate) / downside_vol
            if downside_vol > 1e-12 else 0.0
        )

        metricas = {
            'expected_return': retorno_portafolio,
            'volatility': downside_vol,
            'sharpe_ratio': sortino_final,  # En contexto PMPT, Sharpe ≡ Sortino
        }

        return {
            'weights': {ticker: float(w) for ticker, w in zip(expected_returns.index, pesos)},
            **metricas,
            'sortino_ratio': float(sortino_final),
            'target_return': self.target_return,
            'optimizer_metadata': {
                'method': 'slsqp',
                'success': resultado.success,
                'message': resultado.message,
                'iterations': resultado.nit,
            },
        }
