"""
Optimizador max-Sharpe long-only vía QP exacto, con cascada de respaldo.

Responsabilidad: `optimizar_max_sharpe(mu, sigma, ...)`, el optimizador común de
todas las estrategias del TFM. Usa el cambio de variable de Cornuéjols-Tütüncü
(y = w / (µ'w)): `min y'Σy  s.a.  µ'y = 1, y >= 0`, con w = y / 1'y. Como el
Sharpe es homogéneo de grado 0 en w, la transformación es exacta y el problema
es un QP convexo (óptimo global, sin punto inicial).

Cascada (devuelve siempre `(pesos, degenerate, solver_status)`):
  1. Contingencia: si max(µ - rf) <= 0 → mínima varianza, degenerate=True
     (`"gmv_contingencia"`).
  2. cvxpy QP (OSQP, luego CLARABEL)                       → `"qp"`.
  3. SLSQP sobre el ratio de Sharpe                        → `"slsqp"`.
  4. pypfopt.EfficientFrontier.max_sharpe                  → `"pypfopt"`.
  5. Mínima varianza (SLSQP), degenerate=True              → `"gmv_fallback"`.

Penalización de turnover (extensión respecto al TFM; por defecto desactivada y
entonces el comportamiento es EXACTAMENTE el del original):
con `penalizacion_turnover = λ > 0` y `pesos_previos` se resuelve, en la variable
w, el problema media-varianza
    max_w  µ'w - (δ/2) w'Σw - λ ‖w - w_prev‖₁   s.a. 1'w = 1, 0 <= w <= peso_maximo,
donde δ = µ'w_t / (w_t'Σ w_t) y w_t es la cartera tangente exacta de los pasos 2-4.
Con ese δ la cartera media-varianza sin penalización coincide con w_t (las
condiciones KKT de ambos problemas son equivalentes), así que λ=0 recupera el
max-Sharpe y λ>0 es una regularización continua hacia w_prev. Ya NO es el QP
exacto de Sharpe: el Sharpe de la cartera resultante es <= al de w_t. Estado:
`"qp_turnover"`. Si la contingencia GMV se activa, la penalización no se aplica.

Se relaciona con: estimadores_mu.py (Michaud lo llama por réplica), rejilla.py
(motor walk-forward), portfolio_models.MaxSharpeQPOptimizer (envoltorio Strategy)
y el módulo original del TFM `estrategias_rejilla_v2.py` (del que se porta).
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class _ProblemaQPMaxSharpe:
    """
    Problema QP precompilado (cvxpy) para max-Sharpe long-only: `min y'Σy` con
    `µ'y = 1`, `y >= 0` (y `y <= peso_maximo·1'y` si hay tope de peso).

    Se compila una vez por (n, peso_maximo) con parámetros (L = Cholesky de Σ, µ)
    y se reutiliza con `warm_start`. `resolver` devuelve los pesos normalizados o
    None si algo falla (entradas no finitas, Σ no definida positiva, solver).
    """

    def __init__(self, n: int, peso_maximo: float = 1.0):
        import cvxpy as cp

        self._cp = cp
        self.n = n
        self.L = cp.Parameter((n, n))
        self.mu = cp.Parameter(n)
        self.y = cp.Variable(n, nonneg=True)
        restricciones = [self.mu @ self.y == 1.0]
        if peso_maximo < 1.0:
            restricciones.append(self.y <= peso_maximo * cp.sum(self.y))
        self.problema = cp.Problem(cp.Minimize(cp.sum_squares(self.L.T @ self.y)), restricciones)

    def resolver(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray | None:
        """Resuelve el QP para (µ, Σ); None si no se obtiene una solución válida."""
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
            return None
        try:
            L = np.linalg.cholesky(sigma + 1e-10 * np.eye(self.n))
        except np.linalg.LinAlgError:
            return None
        try:
            self.L.value = L
            self.mu.value = mu
        except (ValueError, TypeError):
            # cvxpy puede rechazar matrices al límite de la singularidad: se cae a la cascada.
            return None
        for solver in (self._cp.OSQP, self._cp.CLARABEL):
            try:
                self.problema.solve(solver=solver, warm_start=True)
                y = self.y.value
                if y is not None and np.all(np.isfinite(y)) and y.sum() > 1e-8:
                    w = np.maximum(y, 0.0)
                    return w / w.sum()
            except Exception:  # noqa: BLE001, S112 - se prueba el siguiente solver
                continue
        return None


# Caché de problemas compilados por (n, peso_maximo) dentro del proceso.
_CACHE_QP: dict[tuple[int, float], _ProblemaQPMaxSharpe | None] = {}


def _obtener_qp(n: int, peso_maximo: float) -> _ProblemaQPMaxSharpe | None:
    """Devuelve (y cachea) el problema QP compilado para (n, peso_maximo); None si cvxpy falla."""
    clave = (n, peso_maximo)
    if clave not in _CACHE_QP:
        try:
            _CACHE_QP[clave] = _ProblemaQPMaxSharpe(n, peso_maximo)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[QP] No se pudo compilar el problema (n=%s): %s", n, exc)
            _CACHE_QP[clave] = None
    return _CACHE_QP[clave]


def optimizar_min_varianza(sigma: np.ndarray, peso_maximo: float = 1.0) -> np.ndarray:
    """Cartera de mínima varianza long-only (SLSQP); regla de contingencia, 1/N si no converge."""
    n = sigma.shape[0]

    def varianza(w):
        return float(w @ sigma @ w)

    def gradiente(w):
        return 2.0 * sigma @ w

    try:
        res = minimize(
            varianza, np.full(n, 1.0 / n), jac=gradiente, method="SLSQP",
            bounds=[(0.0, peso_maximo)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if res.success and not np.any(np.isnan(res.x)):
            pesos = np.maximum(res.x, 0.0)
            return pesos / pesos.sum()
    except Exception:  # noqa: BLE001, S110
        pass
    return np.full(n, 1.0 / n)


def _resolver_con_penalizacion_turnover(
    exceso: np.ndarray,
    sigma: np.ndarray,
    pesos_tangente: np.ndarray,
    pesos_previos: np.ndarray,
    penalizacion: float,
    peso_maximo: float,
) -> np.ndarray | None:
    """
    Media-varianza con penalización L1 al turnover (ver docstring del módulo).
    Devuelve None si no hay solución válida (el llamador conserva la cartera tangente).
    """
    import cvxpy as cp

    varianza_t = float(pesos_tangente @ sigma @ pesos_tangente)
    retorno_t = float(pesos_tangente @ exceso)
    if varianza_t <= 1e-14 or retorno_t <= 0:
        return None
    aversion = retorno_t / varianza_t  # δ: con él, la media-varianza sin penalización da w_t
    try:
        L = np.linalg.cholesky(sigma + 1e-10 * np.eye(len(exceso)))
    except np.linalg.LinAlgError:
        return None
    w = cp.Variable(len(exceso), nonneg=True)
    objetivo = (
        exceso @ w
        - 0.5 * aversion * cp.sum_squares(L.T @ w)
        - penalizacion * cp.norm1(w - pesos_previos)
    )
    problema = cp.Problem(cp.Maximize(objetivo), [cp.sum(w) == 1.0, w <= peso_maximo])
    for solver in (cp.CLARABEL, cp.OSQP):
        try:
            problema.solve(solver=solver)
            if w.value is not None and np.all(np.isfinite(w.value)):
                pesos = np.maximum(w.value, 0.0)
                if pesos.sum() > 1e-8:
                    return pesos / pesos.sum()
        except Exception:  # noqa: BLE001, S112
            continue
    return None


def _max_sharpe_exacto(
    mu: np.ndarray, sigma: np.ndarray, rf: float, peso_maximo: float
) -> tuple[np.ndarray, bool, str]:
    """Cascada contingencia → QP → SLSQP → pypfopt → GMV (sin penalización de turnover)."""
    n = len(mu)
    exceso = mu - rf

    if np.all(exceso <= 0):
        logger.debug("[MaxSharpe] Todos µ <= rf, mínima varianza (contingencia).")
        return optimizar_min_varianza(sigma, peso_maximo), True, "gmv_contingencia"

    qp = _obtener_qp(n, peso_maximo)
    if qp is not None:
        w = qp.resolver(exceso, sigma)
        if w is not None:
            return w, False, "qp"

    def sharpe_negativo(w):
        exc = float(w @ exceso)
        vol = float(np.sqrt(w @ sigma @ w))
        if exc <= 0 or vol < 1e-12:
            return 1e6
        return -exc / vol

    try:
        res = minimize(
            sharpe_negativo, np.full(n, 1.0 / n), method="SLSQP", bounds=[(0.0, peso_maximo)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if res.success and not np.any(np.isnan(res.x)) and res.fun < 0:
            pesos = np.maximum(res.x, 0.0)
            if pesos.sum() > 1e-6:
                return pesos / pesos.sum(), False, "slsqp"
    except Exception:  # noqa: BLE001, S110
        pass

    try:
        from pypfopt import EfficientFrontier

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ef = EfficientFrontier(pd.Series(mu), pd.DataFrame(sigma), weight_bounds=(0, peso_maximo))
            ef.max_sharpe(risk_free_rate=rf)
            pesos = np.array(list(ef.clean_weights().values()), dtype=float)
        if not np.any(np.isnan(pesos)) and pesos.sum() > 0.99:
            return pesos / pesos.sum(), False, "pypfopt"
    except Exception:  # noqa: BLE001, S110
        pass

    logger.debug("[MaxSharpe] Todos los métodos fallaron, mínima varianza.")
    return optimizar_min_varianza(sigma, peso_maximo), True, "gmv_fallback"


def optimizar_max_sharpe(
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float = 0.0,
    peso_maximo: float = 1.0,
    pesos_previos: np.ndarray | None = None,
    penalizacion_turnover: float = 0.0,
) -> tuple[np.ndarray, bool, str]:
    """
    Max-Sharpe long-only con cascada de respaldo (ver docstring del módulo).

    Parámetros
    ----------
    mu : (N,) retornos esperados (anualizados) en la misma unidad que sigma.
    sigma : (N, N) covarianza.
    rf : tasa libre de riesgo en la unidad de `mu` (0 si `mu` ya son excesos).
    peso_maximo : tope de peso por activo.
    pesos_previos : (N,) pesos de partida (a la deriva) para la penalización; opcional.
    penalizacion_turnover : λ >= 0. Con 0 (por defecto) o sin `pesos_previos`, es el
        max-Sharpe exacto del TFM.

    Retorna
    -------
    (pesos, degenerate, solver_status): pesos (N,) que suman 1; `degenerate` True si
    se activó la contingencia de mínima varianza; estado en
    {"qp","slsqp","pypfopt","gmv_contingencia","gmv_fallback","qp_turnover"}.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    pesos, degenerate, estado = _max_sharpe_exacto(mu, sigma, rf, peso_maximo)

    if penalizacion_turnover > 0 and pesos_previos is not None and not degenerate:
        previos = np.asarray(pesos_previos, dtype=float)
        if previos.shape != pesos.shape:
            raise ValueError("pesos_previos debe tener la misma longitud que mu")
        penalizados = _resolver_con_penalizacion_turnover(
            mu - rf, sigma, pesos, previos, float(penalizacion_turnover), peso_maximo
        )
        if penalizados is not None:
            return penalizados, False, "qp_turnover"
        logger.warning("[MaxSharpe] La penalización de turnover no se pudo resolver; se mantiene la tangente.")
    return pesos, degenerate, estado
