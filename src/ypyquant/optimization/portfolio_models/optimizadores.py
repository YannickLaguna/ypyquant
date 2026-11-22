"""
Optimizadores de pesos (Strategy `PortfolioOptimizer`).

Responsabilidad: convertir (µ, Σ, restricciones) en pesos y métricas:
- Clásicos: `MarkowitzOptimizer` (pypfopt, SLSQP de reserva), `RiskParityOptimizer` (delega en
  `estimacion_robusta.riesgo.optimizar_paridad_riesgo`).
- PMPT: `PMPTMinCVaROptimizer` (CVaR vía `estimacion_robusta.riesgo.var_cvar_historicos`),
  `PMPTMaxSortinoOptimizer`.
- Del TFM: `MaxSharpeQPOptimizer`, `MichaudOptimizer` (envoltorios de `estimacion_robusta`, implementación
  canónica de max-Sharpe QP, volatilidad objetivo y remuestreo de Michaud).
- Robustos: `RobustoElipsoideOptimizer`, `RobustoCajaOptimizer` (incertidumbre en µ, `estimacion_robusta.elipsoide`).

Se relaciona con: base.py (interfaz), modelo.py (los compone), estimacion_robusta/ (algoritmos),
optimizador_portafolio.py (funciones scipy heredadas de GrowSave) y presets/.
"""

import logging

import numpy as np
import pandas as pd

from ypyquant.optimization.estimacion_robusta.elipsoide import (
    delta_caja,
    kappa_confianza,
    matriz_incertidumbre,
    optimizar_robusto_caja,
    optimizar_robusto_elipsoide,
    z_confianza,
)
from ypyquant.optimization.estimacion_robusta.estimadores_mu import estimar_mu_michaud
from ypyquant.optimization.estimacion_robusta.max_sharpe_qp import optimizar_max_sharpe, optimizar_volatilidad_objetivo
from ypyquant.optimization.estimacion_robusta.riesgo import optimizar_paridad_riesgo, var_cvar_historicos
from ypyquant.optimization.portfolio_models.base import PortfolioOptimizer

logger = logging.getLogger(__name__)


class MarkowitzOptimizer(PortfolioOptimizer):
    """
    Optimización media-varianza clásica (Markowitz).

    I: μ (Series), Σ (DataFrame), objective='max_sharpe'|'min_volatility'|'efficient_risk'
       (este último con `target_volatility`, σ anual objetivo).
    T: pypfopt.EfficientFrontier con backend CVXPY (fallback SLSQP si pypfopt falla);
       efficient_risk con `optimizar_volatilidad_objetivo` (máx. µ'w con σ <= objetivo).
    O: Pesos óptimos + métricas (retorno esperado, volatilidad, Sharpe).
    """

    def __init__(
        self,
        objective: str = "max_sharpe",
        risk_free_rate: float = 0.02,
        target_volatility: float | None = None,
    ):
        super().__init__(name=f"markowitz_{objective}")
        self.objective = objective
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility
        self.config = {'objective': objective, 'risk_free_rate': risk_free_rate, 'target_volatility': target_volatility}

    def optimize(
        self,
        expected_returns: pd.Series,
        risk_matrix: pd.DataFrame,
        constraints: dict | None = None,
        **kwargs
    ) -> dict:
        self.validate_inputs(expected_returns, risk_matrix)

        max_weight = constraints.get('max_weight', 1.0) if constraints else 1.0

        if self.objective == "efficient_risk":
            if not self.target_volatility:
                raise ValueError("efficient_risk requiere target_volatility (σ anual objetivo, p. ej. 0.12)")
            return _resultado_volatilidad_objetivo(
                self, expected_returns, risk_matrix, self.target_volatility, max_weight, self.risk_free_rate
            )

        # Intentar con pypfopt.EfficientFrontier (backend CVXPY, más estable numéricamente)
        try:
            from pypfopt import EfficientFrontier as _EF

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
        constraints: dict | None = None,
        **kwargs
    ) -> dict:
        self.validate_inputs(expected_returns, risk_matrix)

        max_weight = constraints.get('max_weight', 1.0) if constraints else 1.0
        weights, result = optimizar_paridad_riesgo(risk_matrix, max_weight, maxiter=1000)

        metrics = self.calculate_metrics(weights, expected_returns, risk_matrix)

        return {
            'weights': {ticker: float(w) for ticker, w in zip(expected_returns.index, weights)},
            **metrics,
            'optimizer_metadata': {
                'success': result.success,
                'message': result.message
            }
        }


class PMPTMinCVaROptimizer(PortfolioOptimizer):
    """
    Minimiza el CVaR (Expected Shortfall) empírico del portafolio.

    I: μ, Σ, retornos_historicos (DataFrame, kwarg obligatorio), nivel_confianza.
    T: pypfopt.EfficientCVaR.min_cvar() (fallback SLSQP sobre CVaR empírico).
    O: Pesos + métricas + 'cvar' (pérdida esperada en el peor α% de días).
    """

    requiere_historico = True  # PortfolioOptimizationModel le pasa returns_df como retornos_historicos

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
        _, cvar = var_cvar_historicos(retornos_hist, pesos, self.nivel_confianza)
        return float(cvar)

    def optimize(
        self,
        expected_returns: pd.Series,
        risk_matrix: pd.DataFrame,
        constraints: dict | None = None,
        **kwargs
    ) -> dict:
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
        constraints: dict | None = None,
        **kwargs
    ) -> dict:
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


def _resultado_volatilidad_objetivo(
    optimizador: PortfolioOptimizer, expected_returns: pd.Series, risk_matrix: pd.DataFrame,
    volatilidad_objetivo: float, peso_maximo: float, risk_free_rate: float,
) -> dict:
    """Resuelve máx. µ'w con σ <= objetivo y empaqueta pesos, métricas y metadatos (estado del solver)."""
    pesos, estado = optimizar_volatilidad_objetivo(
        expected_returns.values, risk_matrix.values, volatilidad_objetivo, peso_maximo=peso_maximo
    )
    metricas = optimizador.calculate_metrics(pesos, expected_returns, risk_matrix, risk_free_rate)
    metadatos = {'method': estado, 'success': estado != 'gmv_fallback', 'volatilidad_objetivo': volatilidad_objetivo,
                 'degenerate': estado.startswith('gmv')}
    if estado == 'gmv_volatilidad_inalcanzable':
        metadatos['avisos'] = [(
            f"La volatilidad objetivo ({volatilidad_objetivo:.1%}) está por debajo de la mínima posible con "
            f"estos activos ({metricas['volatility']:.1%}): se devuelve la cartera de mínima varianza. "
            "Para bajar más hay que añadir liquidez."
        )]
    return {
        'weights': {t: float(w) for t, w in zip(expected_returns.index, pesos, strict=True)},
        **metricas,
        'optimizer_metadata': metadatos,
    }


class MaxSharpeQPOptimizer(PortfolioOptimizer):
    """
    Max-Sharpe long-only exacto vía QP de Cornuéjols-Tütüncü, con cascada QP → SLSQP → pypfopt → GMV.

    I: µ, Σ (anualizados), `constraints['max_weight']`, kwarg opcional `pesos_previos`
       (con `penalizacion_turnover` > 0 activa la variante media-varianza con L1).
       Con `volatilidad_objetivo` (σ anual) cambia de objetivo: máx. µ'w con σ <= objetivo.
    T: `optimizar_max_sharpe` u `optimizar_volatilidad_objetivo` (estimacion_robusta.max_sharpe_qp).
    O: pesos + métricas; `optimizer_metadata` incluye `method` (estado del solver) y `degenerate`.
    """

    def __init__(
        self, risk_free_rate: float = 0.0, penalizacion_turnover: float = 0.0,
        volatilidad_objetivo: float | None = None,
    ):
        super().__init__(name="max_sharpe_qp")
        self.risk_free_rate = risk_free_rate
        self.penalizacion_turnover = penalizacion_turnover
        self.volatilidad_objetivo = volatilidad_objetivo
        self.config = {'risk_free_rate': risk_free_rate, 'penalizacion_turnover': penalizacion_turnover,
                       'volatilidad_objetivo': volatilidad_objetivo}

    def optimize(
        self, expected_returns: pd.Series, risk_matrix: pd.DataFrame, constraints: dict | None = None, **kwargs
    ) -> dict:
        self.validate_inputs(expected_returns, risk_matrix)
        peso_maximo = constraints.get('max_weight', 1.0) if constraints else 1.0
        if self.volatilidad_objetivo:
            return _resultado_volatilidad_objetivo(
                self, expected_returns, risk_matrix, self.volatilidad_objetivo, peso_maximo, self.risk_free_rate
            )
        pesos, degenerate, estado = optimizar_max_sharpe(
            expected_returns.values, risk_matrix.values, rf=self.risk_free_rate, peso_maximo=peso_maximo,
            pesos_previos=kwargs.get('pesos_previos'), penalizacion_turnover=self.penalizacion_turnover,
        )
        metricas = self.calculate_metrics(pesos, expected_returns, risk_matrix, self.risk_free_rate)
        return {
            'weights': {t: float(w) for t, w in zip(expected_returns.index, pesos, strict=True)},
            **metricas,
            'optimizer_metadata': {'method': estado, 'degenerate': bool(degenerate), 'success': True},
        }


class MichaudOptimizer(PortfolioOptimizer):
    """
    Remuestreo eficiente de Michaud: bootstrap de la ventana histórica, refit de µ y Σ_LW por
    réplica, max-Sharpe QP en cada una y promedio de pesos.

    I: kwarg `retornos_historicos` (DataFrame T x N, en la frecuencia `frequency`); si se usa vía
       `PortfolioOptimizationModel` se rellena solo con `returns_df`. µ y Σ recibidos solo
       sirven para calcular las métricas del resultado (Michaud reestima los suyos).
    T: `estimar_mu_michaud` con `metodo_bootstrap` "iid" o "bloques" (Politis-Romano); con
       `volatilidad_objetivo` cada réplica resuelve máx. µ'w con σ <= objetivo en vez de max-Sharpe
       (el promedio cumple σ <= objetivo con cada Σ de réplica, por convexidad; con la Σ completa,
       aproximadamente).
    O: pesos promedio + métricas + `optimizer_metadata` con `frac_contingencia` y `degenerate`
       (mayoría de réplicas en contingencia GMV).
    """

    requiere_historico = True

    def __init__(
        self,
        n_replicas: int = 50,
        metodo_bootstrap: str = "iid",
        longitud_media_bloque: float = 6.0,
        frequency: int = 252,
        risk_free_rate: float = 0.0,
        seed: int | None = None,
        volatilidad_objetivo: float | None = None,
    ):
        super().__init__(name="michaud")
        self.n_replicas = n_replicas
        self.metodo_bootstrap = metodo_bootstrap
        self.longitud_media_bloque = longitud_media_bloque
        self.frequency = frequency
        self.risk_free_rate = risk_free_rate
        self.seed = seed
        self.volatilidad_objetivo = volatilidad_objetivo
        self.config = {
            'n_replicas': n_replicas, 'metodo_bootstrap': metodo_bootstrap,
            'longitud_media_bloque': longitud_media_bloque, 'frequency': frequency,
            'risk_free_rate': risk_free_rate, 'seed': seed, 'volatilidad_objetivo': volatilidad_objetivo,
        }

    def optimize(
        self, expected_returns: pd.Series, risk_matrix: pd.DataFrame, constraints: dict | None = None, **kwargs
    ) -> dict:
        historico = kwargs.get('retornos_historicos')
        if historico is None:
            raise ValueError("MichaudOptimizer requiere 'retornos_historicos' como kwarg.")
        self.validate_inputs(expected_returns, risk_matrix)
        historico = historico[list(expected_returns.index)]
        peso_maximo = constraints.get('max_weight', 1.0) if constraints else 1.0

        pesos, frac_contingencia = estimar_mu_michaud(
            historico, n_replicas=self.n_replicas, rf=self.risk_free_rate / self.frequency,
            peso_maximo=peso_maximo, devolver_diagnostico=True, metodo_bootstrap=self.metodo_bootstrap,
            longitud_media_bloque=self.longitud_media_bloque,
            rng=np.random.default_rng(self.seed) if self.seed is not None else None, frecuencia=self.frequency,
            volatilidad_objetivo=self.volatilidad_objetivo,
        )
        metricas = self.calculate_metrics(pesos, expected_returns, risk_matrix, self.risk_free_rate)
        degenerate = bool(frac_contingencia > 0.5) if np.isfinite(frac_contingencia) else False
        return {
            'weights': {t: float(w) for t, w in zip(expected_returns.index, pesos, strict=True)},
            **metricas,
            'optimizer_metadata': {
                'method': 'michaud_bootstrap', 'success': True, 'n_replicas': self.n_replicas,
                'metodo_bootstrap': self.metodo_bootstrap, 'frac_contingencia': float(frac_contingencia),
                'degenerate': degenerate, 'volatilidad_objetivo': self.volatilidad_objetivo,
            },
        }


def _resultado_robusto(
    optimizador: PortfolioOptimizer, expected_returns: pd.Series, risk_matrix: pd.DataFrame,
    pesos: np.ndarray, estado: str, diagnostico: dict, volatilidad_objetivo: float | None,
    risk_free_rate: float, parametros: dict,
) -> dict:
    """Empaqueta pesos, métricas y metadatos (estado del solver, diagnóstico del peor caso y avisos)."""
    metricas = optimizador.calculate_metrics(pesos, expected_returns, risk_matrix, risk_free_rate)
    metadatos = {'method': estado, 'success': estado != 'gmv_fallback', 'degenerate': estado.startswith('gmv'),
                 'volatilidad_objetivo': volatilidad_objetivo, 'diagnostico': diagnostico, **parametros}
    if estado == 'gmv_volatilidad_inalcanzable':
        metadatos['avisos'] = [(
            f"La volatilidad objetivo ({volatilidad_objetivo:.1%}) está por debajo de la mínima posible con "
            f"estos activos ({metricas['volatility']:.1%}): se devuelve la cartera de mínima varianza. "
            "Para bajar más hay que añadir liquidez."
        )]
    return {
        'weights': {t: float(w) for t, w in zip(expected_returns.index, pesos, strict=True)},
        **metricas,
        'optimizer_metadata': metadatos,
    }


def _observaciones_ventana(optimizador: PortfolioOptimizer, kwargs: dict) -> int:
    """T: `n_observaciones` fijado en el optimizador o, si no, la longitud del histórico recibido."""
    if optimizador.n_observaciones:
        return int(optimizador.n_observaciones)
    historico = kwargs.get('retornos_historicos')
    if historico is None:
        raise ValueError(
            f"{type(optimizador).__name__} necesita 'retornos_historicos' (para conocer T) o `n_observaciones`."
        )
    return len(historico)


class RobustoElipsoideOptimizer(PortfolioOptimizer):
    """
    Optimización robusta con incertidumbre elipsoidal en µ (Goldfarb-Iyengar 2003; Ceria-Stubbs 2006):

        max_w  µ'w - κ‖Θ^{1/2}w‖ - (aversión/2) w'Σw   s.a.  1'w=1, 0<=w<=peso_maximo [, σ(w) <= σ*]

    I: µ y Σ anualizados del modelo de rentabilidad/riesgo (como los demás optimizadores),
       `constraints['max_weight']` y el kwarg `retornos_historicos` (lo rellena solo
       `PortfolioOptimizationModel`) para conocer T, la longitud de la ventana de estimación
       (o `n_observaciones` fijo). Θ = frequency·Σ/T (`tipo_theta`="sigma") o su diagonal.
    T: `optimizar_robusto_elipsoide` (SOCP). κ = `kappa` si se da; si no, sqrt(χ²_N(`confianza`)).
       Con `kappa`=0 reproduce media-varianza.
    O: pesos + métricas; `optimizer_metadata` con `method`, `kappa`, `diagnostico`
       (rentabilidad nominal y de peor caso, volatilidad, penalización) y `degenerate`.
    """

    requiere_historico = True

    def __init__(
        self, kappa: float | None = None, confianza: float = 0.9, tipo_theta: str = "sigma",
        aversion: float | None = 2.5, volatilidad_objetivo: float | None = None, frequency: int = 252,
        risk_free_rate: float = 0.0, n_observaciones: int | None = None,
    ):
        super().__init__(name="robusto_elipsoide")
        self.kappa = kappa
        self.confianza = confianza
        self.tipo_theta = tipo_theta
        self.aversion = aversion
        self.volatilidad_objetivo = volatilidad_objetivo
        self.frequency = frequency
        self.risk_free_rate = risk_free_rate
        self.n_observaciones = n_observaciones
        self.config = {
            'kappa': kappa, 'confianza': confianza, 'tipo_theta': tipo_theta, 'aversion': aversion,
            'volatilidad_objetivo': volatilidad_objetivo, 'frequency': frequency,
            'risk_free_rate': risk_free_rate, 'n_observaciones': n_observaciones,
        }

    def optimize(
        self, expected_returns: pd.Series, risk_matrix: pd.DataFrame, constraints: dict | None = None, **kwargs
    ) -> dict:
        self.validate_inputs(expected_returns, risk_matrix)
        peso_maximo = constraints.get('max_weight', 1.0) if constraints else 1.0
        n_obs = _observaciones_ventana(self, kwargs)
        sigma = risk_matrix.values
        theta = matriz_incertidumbre(sigma, n_obs, self.tipo_theta, frecuencia=self.frequency)
        kappa = self.kappa if self.kappa is not None else kappa_confianza(self.confianza, len(expected_returns))
        pesos, estado, diagnostico = optimizar_robusto_elipsoide(
            expected_returns.values, sigma, theta, kappa, aversion=self.aversion,
            volatilidad_objetivo=self.volatilidad_objetivo or None, peso_maximo=peso_maximo,
        )
        return _resultado_robusto(
            self, expected_returns, risk_matrix, pesos, estado, diagnostico, self.volatilidad_objetivo or None,
            self.risk_free_rate, {'kappa': float(kappa), 'tipo_theta': self.tipo_theta, 'n_observaciones': n_obs},
        )


class RobustoCajaOptimizer(PortfolioOptimizer):
    """
    Optimización robusta con incertidumbre de caja |µ_i - µ̂_i| <= δ_i, δ_i = z·σ_i·sqrt(frequency/T):

        max_w  (µ - δ)'w - (aversión/2) w'Σw   s.a.  1'w=1, 0<=w<=peso_maximo [, σ(w) <= σ*]

    I, O: como `RobustoElipsoideOptimizer` (necesita T vía `retornos_historicos` o `n_observaciones`).
    T: `optimizar_robusto_caja`. z = `z` si se da; si no, el cuantil bilateral normal de `confianza`.
       Con z=0 reproduce media-varianza.
    """

    requiere_historico = True

    def __init__(
        self, z: float | None = None, confianza: float = 0.9, aversion: float | None = 2.5,
        volatilidad_objetivo: float | None = None, frequency: int = 252, risk_free_rate: float = 0.0,
        n_observaciones: int | None = None,
    ):
        super().__init__(name="robusto_caja")
        self.z = z
        self.confianza = confianza
        self.aversion = aversion
        self.volatilidad_objetivo = volatilidad_objetivo
        self.frequency = frequency
        self.risk_free_rate = risk_free_rate
        self.n_observaciones = n_observaciones
        self.config = {
            'z': z, 'confianza': confianza, 'aversion': aversion, 'volatilidad_objetivo': volatilidad_objetivo,
            'frequency': frequency, 'risk_free_rate': risk_free_rate, 'n_observaciones': n_observaciones,
        }

    def optimize(
        self, expected_returns: pd.Series, risk_matrix: pd.DataFrame, constraints: dict | None = None, **kwargs
    ) -> dict:
        self.validate_inputs(expected_returns, risk_matrix)
        peso_maximo = constraints.get('max_weight', 1.0) if constraints else 1.0
        n_obs = _observaciones_ventana(self, kwargs)
        sigma = risk_matrix.values
        z = self.z if self.z is not None else z_confianza(self.confianza)
        delta = delta_caja(sigma, n_obs, z, frecuencia=self.frequency)
        pesos, estado, diagnostico = optimizar_robusto_caja(
            expected_returns.values, sigma, delta, aversion=self.aversion,
            volatilidad_objetivo=self.volatilidad_objetivo or None, peso_maximo=peso_maximo,
        )
        return _resultado_robusto(
            self, expected_returns, risk_matrix, pesos, estado, diagnostico, self.volatilidad_objetivo or None,
            self.risk_free_rate, {'z': float(z), 'n_observaciones': n_obs},
        )
