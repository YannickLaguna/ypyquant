"""
Estimadores robustos del vector de retornos esperados µ (TFM «Optimización robusta de portafolios»).

Responsabilidad: funciones puras (DataFrame/ndarray in → ndarray out) que portan
los estimadores de la rejilla v2 del TFM. Todos trabajan con retornos en la
frecuencia indicada (`frecuencia`: 12 mensual por defecto, como el TFM; 252 si
se alimentan retornos diarios) y devuelven µ y Σ anualizados con esa frecuencia.

Estimadores:
- `estimar_mu_muestral`      media aritmética anualizada.
- `estimar_mu_bayes_stein`   Jorion (1986): µ_bs = (1-φ)µ̂ + φ µ_MVG·1.
- `estimar_mu_bl_prior`      Π = δ Σ w_mkt (equilibrio CAPM inverso).
- `calcular_omega_momentum`  Ω_ii = Var(señal)/n_obs de las views de momentum.
- `estimar_mu_bl_momentum`   posterior Black-Litterman con views 12-1 (Woodbury) → (µ, M).
- `calcular_diagnostico_bl`  confianza relativa view/prior de BL-momentum.
- `estimar_mu_michaud`       remuestreo eficiente: bootstrap + refit µ,Σ_LW + promedio de pesos.

Mejoras respecto al original del TFM (todas por defecto compatibles con él):
- Bayes-Stein: el parámetro `si_t_menor_n2` controla qué ocurre cuando T < N+2
  (por defecto aviso; "error" lanza ValueError; "ignorar" es el comportamiento
  silencioso del TFM). La GMV usada para el objetivo del shrinkage es la NO
  restringida (Σ⁻¹1 / 1'Σ⁻¹1, admite pesos negativos), igual que el original y
  que la fórmula de Jorion; no es la cartera long-only del optimizador.
- Michaud: `metodo_bootstrap` = "iid" (por defecto, paridad con el TFM) o
  "bloques" (bootstrap estacionario de Politis-Romano con longitud media de
  bloque `longitud_media_bloque`, que respeta la dependencia serial).
  Además admite un `rng` explícito (np.random.Generator); con `rng=None` usa el
  estado global de numpy como el original (necesario para reproducir sus corridas).

Se relaciona con: sigma.py (Ledoit-Wolf), max_sharpe_qp.py (optimizador de cada
réplica de Michaud), w_mercado.py (provee w_mkt a los estimadores BL),
rejilla.py (los orquesta) y portfolio_models.py (clases Strategy que los envuelven).
"""

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd

from ypyquant.optimization.estimacion_robusta.max_sharpe_qp import optimizar_max_sharpe, optimizar_volatilidad_objetivo
from ypyquant.optimization.estimacion_robusta.sigma import FRECUENCIA_MENSUAL, estimar_sigma_ledoit_wolf

logger = logging.getLogger(__name__)

MESES_MOMENTUM = 12  # ventana de la señal 12-1 (12 meses, se excluye el más reciente)


# ============================================================================
# µ muestral, Bayes-Stein, Black-Litterman
# ============================================================================


def estimar_mu_muestral(retornos: pd.DataFrame, frecuencia: int = FRECUENCIA_MENSUAL) -> np.ndarray:
    """µ̂ = media aritmética de los retornos × `frecuencia` (anualizada). Devuelve (N,)."""
    return retornos.mean().values * frecuencia


def estimar_mu_bayes_stein(
    retornos: pd.DataFrame,
    sigma_lw: pd.DataFrame,
    frecuencia: int = FRECUENCIA_MENSUAL,
    si_t_menor_n2: Literal["avisar", "error", "ignorar"] = "avisar",
) -> np.ndarray:
    """
    Bayes-Stein de Jorion (1986): contrae µ̂ hacia el retorno de la GMV.

        φ = (N+2) / [(N+2) + T·(µ̂-µ_g·1)'Σ⁻¹(µ̂-µ_g·1)],   µ_bs = (1-φ)µ̂ + φ·µ_g·1

    con µ_g = µ̂'w_g y w_g = Σ⁻¹1/(1'Σ⁻¹1) la GMV NO restringida (puede tener pesos
    negativos; es la fórmula de Jorion y lo que hace el original del TFM).

    Escala: µ̂ y Σ vienen anualizados (× frecuencia), así que T entra en AÑOS
    (T/frecuencia); con T en nº de observaciones el denominador quedaría
    inflado ×frecuencia (bug corregido en el TFM v2).

    Condición T >= N+2: Jorion corrige Σ⁻¹ con el factor (T-N-2)/T, que no existe
    si T <= N+2; aquí Σ es Ledoit-Wolf (siempre invertible), pero entonces φ ya no
    es el de Jorion y es dominado por el ruido de Σ. `si_t_menor_n2` decide:
    "avisar" (warnings.warn, defecto), "error" (ValueError) o "ignorar" (silencio,
    comportamiento del TFM, necesario para N=150 con T=12..60).

    Si Σ es singular devuelve µ̂ (con aviso en el log).
    """
    n_observaciones, n_activos = retornos.shape
    if n_observaciones < n_activos + 2:
        mensaje = (
            f"Bayes-Stein con T={n_observaciones} < N+2={n_activos + 2}: la fórmula de Jorion "
            "no está bien definida (Σ⁻¹ sin corrección de muestra finita); φ puede ser poco fiable."
        )
        if si_t_menor_n2 == "error":
            raise ValueError(mensaje)
        if si_t_menor_n2 == "avisar":
            warnings.warn(mensaje, UserWarning, stacklevel=2)

    n_anios = n_observaciones / frecuencia
    mu_hat = retornos.mean().values * frecuencia
    sigma_np = sigma_lw.values

    try:
        unos = np.ones(n_activos)
        sigma_inv_unos = np.linalg.solve(sigma_np, unos)
        w_minvar = sigma_inv_unos / (unos @ sigma_inv_unos)
        mu_min = float(mu_hat @ w_minvar)

        diferencia = mu_hat - mu_min * unos
        distancia = diferencia @ np.linalg.solve(sigma_np, diferencia)
        phi = float(np.clip((n_activos + 2) / ((n_activos + 2) + n_anios * distancia), 0.0, 1.0))
        return (1.0 - phi) * mu_hat + phi * mu_min * unos
    except np.linalg.LinAlgError:
        logger.warning("[BayesStein] Σ singular, devolviendo µ muestral.")
        return mu_hat


def estimar_mu_bl_prior(sigma_lw: pd.DataFrame, w_mkt: np.ndarray, delta: float = 2.5) -> np.ndarray:
    """
    Prior de equilibrio CAPM: Π = δ·Σ·w_mkt.

    Sin views (K=0) Σ⁻¹Π = δ·w_mkt, así que el max-Sharpe sobre (Π, Σ) devuelve
    exactamente w_mkt (y escalar Σ por (1+τ) no cambia los pesos). Sirve para
    diagnóstico y como prior de BL-momentum.
    """
    return delta * sigma_lw.values @ w_mkt


def calcular_omega_momentum(ventana_signal: pd.DataFrame, frecuencia: int = FRECUENCIA_MENSUAL) -> np.ndarray:
    """
    Ω_ii de las views de momentum: error estándar² de la señal (media de la
    ventana), Var(señal)/n_obs anualizada — NO la varianza de las observaciones,
    que con 11 meses es un orden de magnitud mayor y haría ignorar la view.
    """
    n_obs = max(len(ventana_signal), 1)
    varianza_muestra = np.maximum(ventana_signal.var().values * frecuencia, 1e-10)
    return varianza_muestra / n_obs


def estimar_mu_bl_momentum(
    ventana_signal: pd.DataFrame,
    sigma_lw: pd.DataFrame,
    w_mkt: np.ndarray,
    delta: float = 2.5,
    tau: float = 0.05,
    frecuencia: int = FRECUENCIA_MENSUAL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    BL-momentum: posterior He-Litterman con views absolutas de momentum 12-1 (P=I, K=N).

        q_i = media de excesos en t-12..t-2 (excluye t-1),   Ω_ii = Var_i/n_obs

    `ventana_signal` la construye el LLAMADOR por FECHA (no por posición sobre la
    ventana de estimación, que puede tener huecos por dropna de otros activos).

    Posterior vía Woodbury (P=I): K = τΣ(τΣ+Ω)⁻¹, µ_bl = Π + K(q-Π), M = τΣ - KτΣ.

    Retorna (µ_bl, M): µ_bl (N,) anualizado; M (N,N) covarianza posterior de µ, para
    optimizar sobre Σ̄ = Σ + M. Con ventana vacía o sistema singular devuelve (Π, τΣ).
    """
    sigma_np = sigma_lw.values
    pi = delta * sigma_np @ w_mkt

    if ventana_signal.empty:
        logger.warning("[BL-momentum] Ventana de señal vacía, devolviendo (Π, τΣ).")
        return pi, tau * sigma_np

    q = ventana_signal.mean().values * frecuencia
    omega_diag = calcular_omega_momentum(ventana_signal, frecuencia)

    tau_sigma = tau * sigma_np
    try:
        K = tau_sigma @ np.linalg.inv(tau_sigma + np.diag(omega_diag))
    except np.linalg.LinAlgError:
        logger.warning("[BL-momentum] Sistema singular, devolviendo (Π, τΣ).")
        return pi, tau_sigma

    return pi + K @ (q - pi), tau_sigma - K @ tau_sigma


def calcular_diagnostico_bl(
    ventana_signal: pd.DataFrame,
    sigma_lw: pd.DataFrame,
    w_mkt: np.ndarray,
    delta: float = 2.5,
    tau: float = 0.05,
    frecuencia: int = FRECUENCIA_MENSUAL,
) -> dict[str, float]:
    """
    Diagnósticos de la posterior de BL-momentum (cuánto pesa la view frente al prior):

    - `conf_aislada`: n_obs·τ/(1+n_obs·τ), peso de la view en el caso univariante.
    - `conf_conjunta`: mediana transversal del cociente entre el ajuste posterior
      conjunto (usa la correlación de Σ) y el ajuste con confianza total (Ω→0),
      restringida a los activos con |q-Π| >= percentil 10 (evita 0/0).
    - `dispersion_omega`: desviación típica transversal de Ω_ii/(τΣ_ii).

    NaN en los tres si no hay señal; NaN en los dos últimos si el sistema es singular.
    """
    if ventana_signal.empty:
        return {"conf_aislada": np.nan, "conf_conjunta": np.nan, "dispersion_omega": np.nan}

    sigma_np = sigma_lw.values
    n_obs = max(len(ventana_signal), 1)
    conf_aislada = float((n_obs * tau) / (1.0 + n_obs * tau))

    pi = delta * sigma_np @ w_mkt
    q = ventana_signal.mean().values * frecuencia
    omega_diag = calcular_omega_momentum(ventana_signal, frecuencia)
    diferencia = q - pi
    tau_sigma = tau * sigma_np

    try:
        ajuste_conjunto = tau * np.linalg.solve(tau_sigma + np.diag(omega_diag), diferencia)
        ajuste_aislado = np.linalg.solve(sigma_np, diferencia)
    except np.linalg.LinAlgError:
        logger.warning("[Diagnostico BL] Sistema singular, sin conf_conjunta/dispersion_omega.")
        return {"conf_aislada": conf_aislada, "conf_conjunta": np.nan, "dispersion_omega": np.nan}

    abs_dif = np.abs(diferencia)
    mascara = abs_dif >= np.percentile(abs_dif, 10)
    with np.errstate(divide="ignore", invalid="ignore"):
        cociente = np.where(mascara, ajuste_conjunto / ajuste_aislado, np.nan)
    finitos = cociente[np.isfinite(cociente)]
    conf_conjunta = float(np.median(finitos)) if finitos.size else np.nan
    dispersion_omega = float(np.std(omega_diag / (tau * np.diag(sigma_np)))) if tau > 0 else np.nan

    return {"conf_aislada": conf_aislada, "conf_conjunta": conf_conjunta, "dispersion_omega": dispersion_omega}


# ============================================================================
# Michaud: remuestreo eficiente con bootstrap iid o por bloques
# ============================================================================


def _enteros_aleatorios(rng: np.random.Generator | None, alto: int, tamanio: int) -> np.ndarray:
    """Enteros en [0, alto). Con rng=None usa el estado global de numpy (como el original del TFM)."""
    if rng is None:
        return np.random.randint(0, alto, size=tamanio)
    return rng.integers(0, alto, size=tamanio)


def _uniformes_aleatorios(rng: np.random.Generator | None, tamanio: int) -> np.ndarray:
    """U(0,1). Con rng=None usa el estado global de numpy."""
    if rng is None:
        return np.random.random(tamanio)
    return rng.random(tamanio)


def indices_bootstrap(
    n_obs: int,
    metodo: Literal["iid", "bloques"] = "iid",
    longitud_media_bloque: float = 6.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Índices (n_obs,) de una remuestra bootstrap de una serie de `n_obs` filas.

    - "iid": n_obs sorteos con reemplazo (un solo sorteo `randint`, idéntico al TFM).
    - "bloques": bootstrap estacionario de Politis-Romano (1994): la remuestra es una
      concatenación de bloques de longitud geométrica con media `longitud_media_bloque`;
      en cada paso, con probabilidad p=1/L se salta a una posición aleatoria y si no se
      avanza una posición (circular, con envoltura al final). Conserva la dependencia
      serial de corto plazo y la estacionariedad de la remuestra.
    """
    if metodo == "iid":
        return _enteros_aleatorios(rng, n_obs, n_obs)
    if metodo != "bloques":
        raise ValueError(f"metodo_bootstrap desconocido: {metodo!r} (usa 'iid' o 'bloques')")
    if longitud_media_bloque < 1:
        raise ValueError("longitud_media_bloque debe ser >= 1")

    inicios = _enteros_aleatorios(rng, n_obs, n_obs)
    nuevo_bloque = _uniformes_aleatorios(rng, n_obs) < 1.0 / longitud_media_bloque
    indices = np.empty(n_obs, dtype=int)
    indices[0] = inicios[0]
    for i in range(1, n_obs):
        indices[i] = inicios[i] if nuevo_bloque[i] else (indices[i - 1] + 1) % n_obs
    return indices


def estimar_mu_michaud(
    retornos: pd.DataFrame,
    n_replicas: int = 50,
    rf: float = 0.0,
    peso_maximo: float = 1.0,
    devolver_diagnostico: bool = False,
    metodo_bootstrap: Literal["iid", "bloques"] = "iid",
    longitud_media_bloque: float = 6.0,
    rng: np.random.Generator | None = None,
    frecuencia: int = FRECUENCIA_MENSUAL,
    volatilidad_objetivo: float | None = None,
):
    """
    Remuestreo eficiente de Michaud (1989): en cada una de `n_replicas` réplicas se
    remuestrea la ventana (iid o por bloques), se reestiman µ̂^(r) y Σ̂^(r)_LW, se
    resuelve max-Sharpe long-only y se PROMEDIAN LOS PESOS (no los µ).

    Parámetros
    ----------
    retornos : (T x N) retornos de la ventana de estimación.
    rf : tasa libre por periodo (en la frecuencia de `retornos`); si `retornos` ya son
        excesos, 0.
    metodo_bootstrap : "iid" (defecto, paridad con el TFM) o "bloques" (estacionario).
    longitud_media_bloque : L de Politis-Romano (solo para "bloques"); regla práctica
        L ≈ T^(1/3); 6 meses por defecto.
    rng : generador explícito; None usa el estado global de numpy (como el TFM).
    volatilidad_objetivo : σ anual; si se da, cada réplica resuelve máx. µ'w con σ <= objetivo
        (frontera eficiente) en vez de max-Sharpe. La «contingencia» cuenta entonces las
        réplicas que no alcanzan el objetivo y devuelven la GMV.
    devolver_diagnostico : si True devuelve `(pesos, frac_contingencia)` con la fracción
        de réplicas que usaron la contingencia GMV (NaN si ninguna convergió).

    Si ninguna réplica converge devuelve 1/N.
    """
    n_activos = retornos.shape[1]
    n_obs = retornos.shape[0]

    pesos_acumulados = np.zeros(n_activos)
    exitos = 0
    contingencias = 0

    for _ in range(n_replicas):
        try:
            indices = indices_bootstrap(n_obs, metodo_bootstrap, longitud_media_bloque, rng)
            muestra = retornos.iloc[indices]
            mu_r = muestra.mean().values * frecuencia
            sigma_r = estimar_sigma_ledoit_wolf(muestra, frecuencia).values

            if volatilidad_objetivo:
                pesos, estado_r = optimizar_volatilidad_objetivo(
                    mu_r - rf * frecuencia, sigma_r, volatilidad_objetivo, peso_maximo=peso_maximo
                )
                degenerate_r = estado_r.startswith("gmv")
            else:
                pesos, degenerate_r, _ = optimizar_max_sharpe(
                    mu=mu_r - rf * frecuencia, sigma=sigma_r, peso_maximo=peso_maximo
                )
            pesos_acumulados += pesos
            exitos += 1
            if degenerate_r:
                contingencias += 1
        except Exception:  # noqa: BLE001, S112 - una réplica fallida no invalida el promedio (como el TFM)
            continue

    if exitos == 0:
        logger.warning("[Michaud] Ninguna réplica convergió, devolviendo 1/N.")
        pesos_promedio = np.full(n_activos, 1.0 / n_activos)
        frac_contingencia = np.nan
    else:
        pesos_promedio = pesos_acumulados / exitos
        frac_contingencia = contingencias / exitos

    if devolver_diagnostico:
        return pesos_promedio, frac_contingencia
    return pesos_promedio
