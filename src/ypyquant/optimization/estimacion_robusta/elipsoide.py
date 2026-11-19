"""
Optimización robusta de cartera con incertidumbre elipsoidal (y de caja) en µ.

Responsabilidad: funciones puras (sin estado, sin BD) que resuelven el problema de media-varianza
cuando el vector de rentabilidades esperadas µ no se conoce y solo se dispone de la estimación
muestral µ̂. En vez de fiarse de µ̂, se busca la mejor cartera en el PEOR caso de un conjunto de
incertidumbre en torno a µ̂ (Goldfarb e Iyengar, 2003; Ceria y Stubbs, 2006; Fabozzi, Huang y
Zhou, 2010).

Conjunto elipsoidal:   U = { µ : (µ - µ̂)' Θ⁻¹ (µ - µ̂) <= κ² },  Θ = covarianza del estimador de µ.
Como min_{µ∈U} µ'w = µ̂'w - κ‖Θ^{1/2} w‖₂, el problema robusto es un SOCP:

    max_w  µ̂'w - κ‖Θ^{1/2} w‖₂ - (aversion/2) w'Σw     s.a.  1'w = 1,  0 <= w <= peso_maximo
    (opcional)  ‖Lᵀw‖₂ <= σ*   con Σ = LLᵀ   (volatilidad objetivo)

El término κ‖Θ^{1/2}w‖ es una penalización de "riesgo de estimación": castiga las posiciones
grandes en activos cuya µ está mal estimada (Θ grande) y, con Θ ∝ Σ, empuja hacia la mínima
varianza.

Conjunto de caja:      U = { µ : |µ_i - µ̂_i| <= δ_i },  δ_i = z·σ_i/√T.
Con pesos long-only el peor caso es (µ̂ - δ)'w: cada µ̂_i se rebaja por su error típico; el
problema sigue siendo un QP/SOCP.

Escalas. Θ es la covarianza del estimador muestral de µ: Θ = Σ/T con Σ y µ en la frecuencia de
las observaciones. Si Σ y µ están anualizados con `frecuencia` observaciones por año (252 en
datos diarios), la media anualizada es frecuencia·(media diaria) y su covarianza queda
Θ = frecuencia·Σ_anual/T, por eso `matriz_incertidumbre` y `delta_caja` aceptan `frecuencia`
(con frecuencia=1 se recupera Θ = Σ/T literal).

Elección de κ. Con confianza `c`, κ = sqrt(χ²_N(c)) (`kappa_confianza`): el elipsoide contiene
al µ verdadero con probabilidad c bajo normalidad. Con N grande κ crece mucho (χ²_N tiene media N,
así que κ ≈ √N): el conjunto es enorme respecto a la incertidumbre de cada activo y la cartera se
vuelve casi la de mínimo Θ-riesgo. Por eso también se admite pasar κ directamente (un κ pequeño,
p. ej. 0,5–2, actúa como un parámetro de regularización a calibrar, no como un cuantil).

Casos límite (verificados en tests/test_elipsoide.py):
- κ = 0 (o δ = 0): media-varianza clásica; con `volatilidad_objetivo` coincide con
  `optimizar_volatilidad_objetivo`.
- κ → ∞: cartera de mínimo ‖Θ^{1/2}w‖; con Θ ∝ Σ es la de mínima varianza global (GMV).

Estados devueltos: "robusto_elipsoide" / "robusto_caja" (sin objetivo de volatilidad),
"volatilidad_objetivo" (restricción activa), "volatilidad_objetivo_holgada" (σ < σ*),
"gmv_volatilidad_inalcanzable" (σ* por debajo de la mínima varianza; se devuelve la GMV, igual
que `optimizar_volatilidad_objetivo`) y "gmv_fallback" (ningún solver resolvió).

Se relaciona con: max_sharpe_qp.py (reutiliza `optimizar_min_varianza` y los mismos estados y
cascada de solvers CLARABEL → ECOS → SCS), estimadores_mu.py (aporta µ̂), sigma.py (aporta Σ),
portfolio_models.RobustoElipsoideOptimizer / RobustoCajaOptimizer (envoltorios Strategy),
presets/robusta.py (preset `robusta_elipsoide`) y estrategias/incluidas.py (`robusta_elipsoide`,
`robusta_caja`).

Referencias:
- Goldfarb, D. e Iyengar, G. (2003). Robust Portfolio Selection Problems. Mathematics of
  Operations Research 28(1), 1-38.
- Ceria, S. y Stubbs, R. A. (2006). Incorporating estimation errors into portfolio selection:
  Robust portfolio construction. Journal of Asset Management 7(2), 109-127.
- Fabozzi, F. J., Huang, D. y Zhou, G. (2010). Robust portfolios: contributions from operations
  research and finance. Annals of Operations Research 176, 191-220.
"""

import logging
from typing import Literal

import numpy as np
from scipy.stats import chi2, norm

from ypyquant.optimization.estimacion_robusta.max_sharpe_qp import optimizar_min_varianza

logger = logging.getLogger(__name__)

TipoTheta = Literal["sigma", "diagonal"]

_SOLVERS = ("CLARABEL", "ECOS", "SCS")


# ---------------------------------------------------------------------------
# Construcción de la incertidumbre
# ---------------------------------------------------------------------------


def matriz_incertidumbre(
    sigma: np.ndarray,
    n_observaciones: int,
    tipo: TipoTheta = "sigma",
    frecuencia: int = 1,
) -> np.ndarray:
    """
    Θ: covarianza del estimador muestral de µ (ver docstring del módulo sobre escalas).

    - tipo="sigma":    Θ = frecuencia·Σ/T (respeta las correlaciones entre errores de µ̂).
    - tipo="diagonal": Θ = frecuencia·diag(Σ)/T (errores de µ̂ independientes; elipsoide alineado
      con los ejes, más barato y menos sensible a una Σ mal estimada).

    Parámetros: `sigma` (N, N); `n_observaciones` T > 0 (longitud de la ventana de estimación);
    `frecuencia` observaciones por año si Σ está anualizada (1 si Σ y µ están en la misma
    frecuencia que las observaciones).
    """
    sigma = _validar_sigma(sigma)
    if n_observaciones <= 0:
        raise ValueError("n_observaciones debe ser > 0")
    if frecuencia <= 0:
        raise ValueError("frecuencia debe ser > 0")
    if tipo == "sigma":
        theta = sigma
    elif tipo == "diagonal":
        theta = np.diag(np.diag(sigma))
    else:
        raise ValueError("tipo debe ser 'sigma' o 'diagonal'")
    return theta * (frecuencia / float(n_observaciones))


def kappa_confianza(confianza: float, n_activos: int) -> float:
    """
    κ = sqrt(χ²_N(confianza)): radio del elipsoide que contiene a µ con probabilidad `confianza`
    bajo normalidad (`scipy.stats.chi2.ppf`).

    Con N grande κ crece mucho (≈ √N, p. ej. N=100 y 90 % dan κ ≈ 10,7): el elipsoide es enorme
    y la cartera robusta se vuelve casi la de mínimo Θ-riesgo. Por eso las funciones del módulo
    aceptan también un κ directo, que suele calibrarse como parámetro de regularización.
    `confianza` = 0 da κ = 0 (sin robustez).
    """
    if not 0.0 <= confianza < 1.0:
        raise ValueError("confianza debe estar en [0, 1)")
    if n_activos < 1:
        raise ValueError("n_activos debe ser >= 1")
    return float(np.sqrt(chi2.ppf(confianza, df=n_activos)))


def z_confianza(confianza: float) -> float:
    """z bilateral de la normal estándar: P(|Z| <= z) = confianza (z = 0 con confianza 0)."""
    if not 0.0 <= confianza < 1.0:
        raise ValueError("confianza debe estar en [0, 1)")
    return float(norm.ppf(0.5 + confianza / 2.0))


def delta_caja(sigma: np.ndarray, n_observaciones: int, z: float, frecuencia: int = 1) -> np.ndarray:
    """δ_i = z·σ_i·sqrt(frecuencia/T): semiancho de la caja de µ_i (error típico de la media por z)."""
    sigma = _validar_sigma(sigma)
    if n_observaciones <= 0 or frecuencia <= 0:
        raise ValueError("n_observaciones y frecuencia deben ser > 0")
    if z < 0:
        raise ValueError("z debe ser >= 0")
    return z * np.sqrt(np.clip(np.diag(sigma), 0.0, None) * frecuencia / float(n_observaciones))


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _validar_sigma(sigma: np.ndarray) -> np.ndarray:
    """Comprueba que Σ sea cuadrada, finita y simétrica; la devuelve como float."""
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("La matriz de covarianza debe ser cuadrada")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("La matriz de covarianza contiene valores no finitos")
    if not np.allclose(sigma, sigma.T, rtol=1e-8, atol=1e-12):
        raise ValueError("La matriz de covarianza debe ser simétrica")
    return sigma


def raiz_semidefinida(matriz: np.ndarray) -> np.ndarray:
    """
    R con RᵀR = M (M simétrica semidefinida positiva), de modo que ‖R w‖₂² = w'Mw.

    Cholesky con un jitter relativo mínimo; si M es singular o indefinida por redondeo se usa la
    descomposición espectral (autovalores negativos recortados a 0), que sirve para semidefinidas.
    """
    n = matriz.shape[0]
    escala = max(float(np.mean(np.abs(np.diag(matriz)))), 1e-300)
    try:
        return np.linalg.cholesky(matriz + 1e-12 * escala * np.eye(n)).T
    except np.linalg.LinAlgError:
        valores, vectores = np.linalg.eigh((matriz + matriz.T) / 2.0)
        return np.sqrt(np.clip(valores, 0.0, None))[:, None] * vectores.T


def _validar_comunes(mu, sigma, aversion, volatilidad_objetivo, peso_maximo):
    """Valida µ, Σ, aversión, σ* y tope de peso; devuelve (mu, sigma, aversion, volatilidad_objetivo)."""
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 1 or len(mu) == 0:
        raise ValueError("mu debe ser un vector (N,) no vacío")
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu contiene valores no finitos")
    sigma = _validar_sigma(sigma)
    if sigma.shape[0] != len(mu):
        raise ValueError(f"Dimensiones incompatibles: mu tiene {len(mu)} activos y sigma {sigma.shape[0]}")
    aversion = 0.0 if aversion is None else float(aversion)
    if aversion < 0:
        raise ValueError("aversion debe ser >= 0")
    if volatilidad_objetivo is not None and volatilidad_objetivo <= 0:
        raise ValueError("volatilidad_objetivo debe ser > 0 (o None)")
    if not 0.0 < peso_maximo <= 1.0:
        raise ValueError("peso_maximo debe estar en (0, 1]")
    if peso_maximo * len(mu) < 1.0 - 1e-12:
        raise ValueError(f"peso_maximo={peso_maximo} es infactible con {len(mu)} activos (suma máxima < 1)")
    return mu, sigma, aversion, volatilidad_objetivo


def _resolver(
    mu: np.ndarray,
    sigma: np.ndarray,
    aversion: float,
    volatilidad_objetivo: float | None,
    peso_maximo: float,
    construir_penalizacion,
    estado_base: str,
    tolerancia: float = 1e-6,
) -> tuple[np.ndarray, str]:
    """
    Núcleo común (cvxpy): max µ'w - penalización(w) - (aversión/2)w'Σw con las restricciones
    long-only, tope de peso y, si procede, ‖Lᵀw‖ <= σ*. `construir_penalizacion(w)` devuelve la
    expresión cvxpy del término robusto (κ‖Θ^{1/2}w‖ o δ'w). Devuelve (pesos, estado).
    """
    import cvxpy as cp

    n = len(mu)
    raiz_sigma = raiz_semidefinida(sigma)
    gmv = optimizar_min_varianza(sigma, peso_maximo)
    if volatilidad_objetivo is not None and float(np.sqrt(gmv @ sigma @ gmv)) > volatilidad_objetivo + tolerancia:
        return gmv, "gmv_volatilidad_inalcanzable"

    w = cp.Variable(n)
    objetivo = mu @ w - construir_penalizacion(w)
    if aversion > 0:
        objetivo = objetivo - 0.5 * aversion * cp.sum_squares(raiz_sigma @ w)
    restricciones = [cp.sum(w) == 1, w >= 0]
    if peso_maximo < 1.0:
        restricciones.append(w <= peso_maximo)
    if volatilidad_objetivo is not None:
        restricciones.append(cp.norm(raiz_sigma @ w, 2) <= volatilidad_objetivo)
    problema = cp.Problem(cp.Maximize(objetivo), restricciones)

    for solver in _SOLVERS:
        try:
            problema.solve(solver=solver)
        except Exception:  # noqa: BLE001, S112 - se prueba el siguiente solver
            continue
        if problema.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            pesos = np.clip(np.asarray(w.value, dtype=float), 0.0, None)
            if pesos.sum() <= 1e-8:
                continue
            pesos /= pesos.sum()
            if volatilidad_objetivo is None:
                return pesos, estado_base
            vol = float(np.sqrt(pesos @ sigma @ pesos))
            activa = vol >= volatilidad_objetivo * (1 - 1e-3)
            return pesos, "volatilidad_objetivo" if activa else "volatilidad_objetivo_holgada"
    logger.warning("[Robusto] Ningún solver resolvió el problema; se devuelve la GMV.")
    return gmv, "gmv_fallback"


def _diagnostico(
    pesos: np.ndarray, mu: np.ndarray, sigma: np.ndarray, penalizacion: float, aversion: float
) -> dict[str, float]:
    """Rentabilidad nominal y de peor caso, volatilidad y penalización de la cartera."""
    nominal = float(mu @ pesos)
    return {
        "rentabilidad_nominal": nominal,
        "rentabilidad_peor_caso": nominal - penalizacion,
        "volatilidad": float(np.sqrt(max(pesos @ sigma @ pesos, 0.0))),
        "penalizacion": float(penalizacion),
        "aversion": float(aversion),
    }


# ---------------------------------------------------------------------------
# Optimizadores públicos
# ---------------------------------------------------------------------------


def optimizar_robusto_elipsoide(
    mu: np.ndarray,
    sigma: np.ndarray,
    theta: np.ndarray,
    kappa: float,
    aversion: float | None = None,
    volatilidad_objetivo: float | None = None,
    peso_maximo: float = 1.0,
) -> tuple[np.ndarray, str, dict[str, float]]:
    """
    Cartera long-only robusta con incertidumbre elipsoidal en µ (SOCP, ver docstring del módulo):

        max_w  µ̂'w - κ‖Θ^{1/2}w‖₂ - (aversion/2) w'Σw   s.a.  1'w=1, 0<=w<=peso_maximo
        [y ‖Lᵀw‖₂ <= σ*  si `volatilidad_objetivo`]

    Parámetros
    ----------
    mu : (N,) estimación µ̂ (misma frecuencia que `sigma` y `theta`).
    sigma : (N, N) covarianza de los retornos (simétrica, semidefinida positiva).
    theta : (N, N) covarianza del estimador de µ (ver `matriz_incertidumbre`).
    kappa : radio κ >= 0 del elipsoide (ver `kappa_confianza`); 0 = sin robustez.
    aversion : λ >= 0 del término de varianza; None equivale a 0 (útil con `volatilidad_objetivo`).
    volatilidad_objetivo : σ* en la unidad de `sigma`; None = sin restricción de volatilidad.
    peso_maximo : tope de peso por activo.

    Retorna
    -------
    (pesos, estado, diagnostico). `estado` y `diagnostico` (rentabilidad_nominal,
    rentabilidad_peor_caso, volatilidad, penalizacion = κ‖Θ^{1/2}w‖) se describen en el módulo.
    """
    mu, sigma, aversion, volatilidad_objetivo = _validar_comunes(
        mu, sigma, aversion, volatilidad_objetivo, peso_maximo
    )
    if kappa is None or kappa < 0 or not np.isfinite(kappa):
        raise ValueError("kappa debe ser un número >= 0")
    theta = _validar_sigma(theta)
    if theta.shape != sigma.shape:
        raise ValueError("theta debe tener las mismas dimensiones que sigma")
    raiz_theta = raiz_semidefinida(theta)

    import cvxpy as cp

    def penalizacion_elipsoide(w):
        return kappa * cp.norm(raiz_theta @ w, 2)

    pesos, estado = _resolver(
        mu, sigma, aversion, volatilidad_objetivo, peso_maximo, penalizacion_elipsoide, "robusto_elipsoide"
    )
    penalizacion = float(kappa * np.linalg.norm(raiz_theta @ pesos))
    return pesos, estado, _diagnostico(pesos, mu, sigma, penalizacion, aversion)


def optimizar_robusto_caja(
    mu: np.ndarray,
    sigma: np.ndarray,
    delta: np.ndarray | float,
    aversion: float | None = None,
    volatilidad_objetivo: float | None = None,
    peso_maximo: float = 1.0,
) -> tuple[np.ndarray, str, dict[str, float]]:
    """
    Cartera long-only robusta con incertidumbre de caja |µ_i - µ̂_i| <= δ_i:

        max_w  (µ̂ - δ)'w - (aversion/2) w'Σw   s.a.  1'w=1, 0<=w<=peso_maximo
        [y ‖Lᵀw‖₂ <= σ*  si `volatilidad_objetivo`]

    Con w >= 0 el peor caso de la caja es µ_i = µ̂_i - δ_i, así que el problema es un QP (SOCP con
    objetivo de volatilidad). `delta` es un vector (N,) >= 0 o un escalar común (ver `delta_caja`).
    Mismas opciones y estados que `optimizar_robusto_elipsoide`; `penalizacion` del diagnóstico es δ'w.
    """
    mu, sigma, aversion, volatilidad_objetivo = _validar_comunes(
        mu, sigma, aversion, volatilidad_objetivo, peso_maximo
    )
    delta = np.broadcast_to(np.asarray(delta, dtype=float), mu.shape).copy()
    if not np.all(np.isfinite(delta)) or np.any(delta < 0):
        raise ValueError("delta debe ser finito y >= 0")

    pesos, estado = _resolver(
        mu, sigma, aversion, volatilidad_objetivo, peso_maximo, lambda w: delta @ w, "robusto_caja"
    )
    return pesos, estado, _diagnostico(pesos, mu, sigma, float(delta @ pesos), aversion)
