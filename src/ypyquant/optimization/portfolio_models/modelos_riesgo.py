"""
Modelos de riesgo Σ (Strategy `RiskModel`).

Responsabilidad: estimar la matriz de covarianza (o semicovarianza) anualizada:
`SampleCovarianceModel`, `LedoitWolfShrinkageModel`, `ExponentialCovarianceModel`, `SemiCovarianceModel`,
`LedoitWolfSemiCovarianceModel` y `PCADenoisedRiskModel`. El shrinkage Ledoit-Wolf coincide numéricamente con
`estimacion_robusta.sigma.estimar_sigma_ledoit_wolf` (misma llamada a pypfopt; lo comprueba
tests/test_optimizacion_caracterizacion.py).

Se relaciona con: base.py (interfaz), modelo.py (los usa), walk_forward.py (elige Σ por experimento) y presets/.
"""

import logging

import numpy as np
import pandas as pd

from ypyquant.optimization.portfolio_models.base import RiskModel

logger = logging.getLogger(__name__)


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
