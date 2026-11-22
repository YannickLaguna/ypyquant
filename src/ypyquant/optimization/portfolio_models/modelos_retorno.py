"""
Modelos de retorno esperado µ (Strategy `ExpectedReturnsModel`).

Responsabilidad: estimar µ anualizado a partir de la matriz de retornos:
- `HistoricalReturnsModel`, `CAPMReturnsModel`, `BlackLittermanReturnsModel` (clásicos, pypfopt/QuantLib).
- `LSTMReturnsModel` (ML, torch opcional).
- `BayesSteinReturnsModel`, `BLPriorReturnsModel`, `BLMomentumReturnsModel`: envoltorios de los estimadores
  del TFM, cuya lógica vive en `estimacion_robusta.estimadores_mu` (implementación canónica).

Se relaciona con: base.py (interfaz), estimacion_robusta/ (lógica de los modelos del TFM),
black_litterman.py (posterior con views), modelo.py (los usa) y presets/.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from ypyquant.optimization.estimacion_robusta.estimadores_mu import (
    estimar_mu_bayes_stein,
    estimar_mu_bl_momentum,
    estimar_mu_bl_prior,
)
from ypyquant.optimization.estimacion_robusta.sigma import estimar_sigma_ledoit_wolf
from ypyquant.optimization.estimacion_robusta.w_mercado import CapitalizacionEstatica, pesos_mercado
from ypyquant.optimization.portfolio_models.base import ExpectedReturnsModel

logger = logging.getLogger(__name__)


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
        market_returns: pd.Series | None = None,
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
        market_caps: dict[str, float] | None = None
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
        cov_matrix: pd.DataFrame | None = None,
        views: dict[str, float] | None = None,
        view_confidences: list[float] | None = None,
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


class LSTMReturnsModel(ExpectedReturnsModel):
    """Modelo de retornos esperados usando LSTM"""

    def __init__(
        self,
        sequence_length: int = 60,
        forecast_horizon: int = 5,
        lstm_units: int = 50,
        dropout: float = 0.2,
        pretrained_models_path: str | None = None
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


def _covarianza_para_modelo(returns_df: pd.DataFrame, cov_matrix: pd.DataFrame | None, frecuencia: int) -> pd.DataFrame:
    """Σ Ledoit-Wolf anualizada si el llamador no aporta `cov_matrix`."""
    if cov_matrix is not None:
        return cov_matrix
    return estimar_sigma_ledoit_wolf(returns_df, frecuencia)


class BayesSteinReturnsModel(ExpectedReturnsModel):
    """
    µ de Bayes-Stein (Jorion 1986): contrae la media muestral hacia el retorno de la GMV.

    I: retornos (T x N) en la frecuencia `frequency` (252 diaria, 12 mensual); opcional `cov_matrix`
       anualizada (si falta se estima Ledoit-Wolf con la misma frecuencia).
    T: φ = (N+2)/((N+2)+T_años·d²) con d² la distancia de Mahalanobis a la GMV NO restringida.
    O: pd.Series de µ anualizados. Si T < N+2 avisa (o lanza con `si_t_menor_n2="error"`).
    """

    def __init__(self, frequency: int = 252, si_t_menor_n2: str = "avisar"):
        super().__init__(name="bayes_stein")
        self.frequency = frequency
        self.si_t_menor_n2 = si_t_menor_n2
        self.config = {'frequency': frequency, 'si_t_menor_n2': si_t_menor_n2}

    def calculate(self, returns_df: pd.DataFrame, cov_matrix: pd.DataFrame | None = None, **kwargs) -> pd.Series:
        self.validate_returns(returns_df)
        sigma = _covarianza_para_modelo(returns_df, cov_matrix, self.frequency)
        mu = estimar_mu_bayes_stein(returns_df, sigma, self.frequency, self.si_t_menor_n2)
        self.metadata['calculation_date'] = datetime.now().isoformat()
        return pd.Series(mu, index=returns_df.columns)


class _ModeloConPesosMercado(ExpectedReturnsModel):
    """
    Base de los modelos Black-Litterman del TFM: resuelve w_mkt con la cadena
    proveedor de capitalización → `market_caps` (dict) → proxy por $-volumen → equiponderado.

    kwargs de `calculate` reconocidos: `market_caps` (dict ticker→capitalización),
    `proveedor_capitalizacion`, `precios` y `volumen` (para el proxy), `fecha`.
    Si no se aporta capitalización real se deja un aviso en `metadata['avisos']`, que
    `PortfolioOptimizationModel.optimize` copia a `optimizer_metadata['avisos']`.
    """

    def __init__(self, name: str, market_caps: dict[str, float] | None = None):
        super().__init__(name=name)
        self.market_caps = market_caps or {}

    def _resolver_pesos_mercado(self, returns_df: pd.DataFrame, **kwargs) -> np.ndarray:
        proveedor = kwargs.get('proveedor_capitalizacion')
        capitalizaciones = kwargs.get('market_caps') or self.market_caps
        if proveedor is None and capitalizaciones:
            proveedor = CapitalizacionEstatica(capitalizaciones)
        fecha = kwargs.get('fecha')
        fecha = returns_df.index[-1] if fecha is None and isinstance(returns_df.index, pd.DatetimeIndex) else fecha
        pesos, fuente = pesos_mercado(
            list(returns_df.columns), pd.Timestamp(fecha) if fecha is not None else pd.Timestamp.now(),
            proveedor, kwargs.get('precios'), kwargs.get('volumen'),
        )
        self.metadata['w_mkt_fuente'] = fuente
        self.metadata['avisos'] = []
        if fuente != "capitalizacion":
            aviso = (
                f"market_caps no disponibles: w_mkt calculado con la fuente '{fuente}' "
                "(el proxy por volumen no es capitalización de mercado)."
            )
            self.metadata['avisos'].append(aviso)
            logger.warning("[%s] %s", self.name, aviso)
        return pesos


class BLPriorReturnsModel(_ModeloConPesosMercado):
    """
    BL-prior (sin views): µ = Π = δ·Σ·w_mkt.

    Con un optimizador max-Sharpe el resultado es w = w_mkt exacto (escalar Σ por (1+τ) no
    altera los pesos). I: retornos (para Σ y, si no hay datos de mercado, equiponderado),
    `market_caps`/`proveedor_capitalizacion`. O: pd.Series de Π anualizado.
    """

    def __init__(self, risk_aversion: float = 2.5, frequency: int = 252, market_caps: dict[str, float] | None = None):
        super().__init__(name="bl_prior", market_caps=market_caps)
        self.risk_aversion = risk_aversion
        self.frequency = frequency
        self.config = {'risk_aversion': risk_aversion, 'frequency': frequency, 'has_market_caps': bool(market_caps)}

    def calculate(self, returns_df: pd.DataFrame, cov_matrix: pd.DataFrame | None = None, **kwargs) -> pd.Series:
        self.validate_returns(returns_df)
        sigma = _covarianza_para_modelo(returns_df, cov_matrix, self.frequency)
        w_mkt = self._resolver_pesos_mercado(returns_df, **kwargs)
        self.pesos_mercado_ = pd.Series(w_mkt, index=returns_df.columns)
        return pd.Series(estimar_mu_bl_prior(sigma, w_mkt, self.risk_aversion), index=returns_df.columns)


class BLMomentumReturnsModel(_ModeloConPesosMercado):
    """
    BL-momentum: posterior de Black-Litterman con views absolutas de momentum 12-1.

    La señal es la media de los retornos de las últimas `ventana_senal` filas EXCLUYENDO las
    `omitir_ultimas` más recientes (12 y 1 por defecto: meses t-12..t-2 con datos mensuales).
    Con datos diarios ajusta ambos parámetros a observaciones (p. ej. 252 y 21).
    `calculate` devuelve µ posterior y deja la covarianza posterior M en `covarianza_posterior_`;
    la TFM optimiza sobre Σ+M: usa `covarianza_predictiva(sigma)`.
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        frequency: int = 252,
        ventana_senal: int = 12,
        omitir_ultimas: int = 1,
        market_caps: dict[str, float] | None = None,
    ):
        super().__init__(name="bl_momentum", market_caps=market_caps)
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.frequency = frequency
        self.ventana_senal = ventana_senal
        self.omitir_ultimas = omitir_ultimas
        self.covarianza_posterior_: pd.DataFrame | None = None
        self.config = {
            'risk_aversion': risk_aversion, 'tau': tau, 'frequency': frequency,
            'ventana_senal': ventana_senal, 'omitir_ultimas': omitir_ultimas, 'has_market_caps': bool(market_caps),
        }

    def calculate(self, returns_df: pd.DataFrame, cov_matrix: pd.DataFrame | None = None, **kwargs) -> pd.Series:
        self.validate_returns(returns_df)
        sigma = _covarianza_para_modelo(returns_df, cov_matrix, self.frequency)
        w_mkt = self._resolver_pesos_mercado(returns_df, **kwargs)
        fin = len(returns_df) - self.omitir_ultimas
        ventana = returns_df.iloc[max(fin - (self.ventana_senal - self.omitir_ultimas), 0):fin]
        if len(ventana) < self.ventana_senal - self.omitir_ultimas:
            self.metadata.setdefault('avisos', []).append(
                f"histórico corto: la señal usa {len(ventana)} observaciones "
                f"(se esperaban {self.ventana_senal - self.omitir_ultimas})."
            )
        mu, posterior = estimar_mu_bl_momentum(
            ventana, sigma, w_mkt, self.risk_aversion, self.tau, self.frequency
        )
        self.covarianza_posterior_ = pd.DataFrame(posterior, index=returns_df.columns, columns=returns_df.columns)
        return pd.Series(mu, index=returns_df.columns)

    def covarianza_predictiva(self, sigma: pd.DataFrame) -> pd.DataFrame:
        """Σ + M (covarianza predictiva) con la M de la última llamada a `calculate`."""
        if self.covarianza_posterior_ is None:
            raise RuntimeError("Llama a calculate() antes de pedir la covarianza predictiva.")
        return sigma + self.covarianza_posterior_
