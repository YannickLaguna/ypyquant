"""
test_optimizacion_caracterizacion.py — Tests de caracterización del núcleo de optimización.

Responsabilidad:
    Medir numéricamente si las implementaciones paralelas de un mismo algoritmo (max-Sharpe,
    mínima varianza, volatilidad objetivo, paridad de riesgo, Ledoit-Wolf, Bayes-Stein, Michaud,
    BL-prior, CVaR, semicovarianza) devuelven lo mismo. Hay dos tipos de aserciones:
      - EQUIVALENCIA EXACTA (atol ~1e-10): las implementaciones coinciden; una puede delegar en la
        otra sin cambiar resultados.
      - DIFERENCIA ACOTADA (cotas medidas el 2026-11-21): las implementaciones NO coinciden (SLSQP
        frente a QP exacto, pypfopt con `clean_weights`, semicovarianza de pypfopt frente a la de
        ceros); la cota documenta cuánto y fija que no se amplíe sin darse cuenta. Delegar una en
        otra cambiaría resultados de GrowSave: es una decisión pendiente (ver
        docs/planes/2026-11-21-optimizacion-canonica.md).

Datos sintéticos y semillas fijas; sin red ni BD.

Se relaciona con: optimization/optimizador_portafolio.py (funciones scipy y servicio),
optimization/portfolio_models* (clases Strategy), optimization/estimacion_robusta/ (núcleo canónico).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ypyquant.optimization import optimizador_portafolio as legado
from ypyquant.optimization import portfolio_models as modelos
from ypyquant.optimization.estimacion_robusta import (
    estimar_mu_bayes_stein,
    estimar_mu_bl_prior,
    estimar_mu_michaud,
    estimar_sigma_ledoit_wolf,
    optimizar_max_sharpe,
    optimizar_min_varianza,
    optimizar_volatilidad_objetivo,
)

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
SEMILLAS = [1, 2, 3]


def _retornos(semilla: int, n: int = 500) -> pd.DataFrame:
    """Retornos diarios con un factor común y medias distintas por activo."""
    rng = np.random.default_rng(semilla)
    factor = rng.normal(0, 0.01, n)
    datos = {
        t: 0.0002 * (i + 1) + 0.5 * factor + rng.normal(0, 0.008 + 0.003 * i, n)
        for i, t in enumerate(TICKERS)
    }
    return pd.DataFrame(datos, index=pd.bdate_range("2022-01-03", periods=n))


def _dif(a, b) -> float:
    """Diferencia máxima absoluta entre dos vectores/matrices."""
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


@pytest.fixture(autouse=True)
def _sin_avisos():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------------------------
# Equivalencias exactas: ya comparten implementación (o son el mismo cálculo)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("semilla", SEMILLAS)
def test_ledoit_wolf_modelo_igual_a_funcion_canonica(semilla):
    r = _retornos(semilla)
    assert _dif(modelos.LedoitWolfShrinkageModel(252).calculate(r), estimar_sigma_ledoit_wolf(r, 252)) < 1e-12


@pytest.mark.parametrize("semilla", SEMILLAS)
def test_bayes_stein_clase_igual_a_funcion_canonica(semilla):
    r = _retornos(semilla)
    sigma = estimar_sigma_ledoit_wolf(r, 252)
    assert _dif(modelos.BayesSteinReturnsModel(252).calculate(r), estimar_mu_bayes_stein(r, sigma, 252)) < 1e-12


@pytest.mark.parametrize("semilla", SEMILLAS)
def test_bl_prior_clase_y_modelo_bl_igual_a_funcion_canonica(semilla):
    r = _retornos(semilla)
    sigma = estimar_sigma_ledoit_wolf(r, 252)
    caps = dict(zip(TICKERS, [5.0, 4.0, 3.0, 2.0, 1.0], strict=True))
    w = np.array(list(caps.values())) / sum(caps.values())
    esperado = estimar_mu_bl_prior(sigma, w, 2.5)
    assert _dif(modelos.BLPriorReturnsModel(2.5, 252, caps).calculate(r), esperado) < 1e-12
    # El prior de `BlackLittermanReturnsModel` (pypfopt) es el mismo δ·Σ·w con la Σ que se le pasa.
    assert _dif(modelos.BlackLittermanReturnsModel(2.5, 0.025, caps).calculate(r, cov_matrix=sigma), esperado) < 1e-12


@pytest.mark.parametrize("semilla", SEMILLAS)
def test_max_sharpe_qp_clase_igual_a_funcion_canonica(semilla):
    r = _retornos(semilla)
    mu, cov = r.mean() * 252, r.cov() * 252
    esperado, _, _ = optimizar_max_sharpe(mu.values, cov.values, 0.01, 0.4)
    res = modelos.MaxSharpeQPOptimizer(risk_free_rate=0.01).optimize(mu, cov, {"max_weight": 0.4})
    assert _dif(list(res["weights"].values()), esperado) < 1e-12


@pytest.mark.parametrize("semilla", SEMILLAS)
def test_michaud_clase_igual_a_funcion_canonica(semilla):
    r = _retornos(semilla, 300)
    mu, cov = r.mean() * 252, r.cov() * 252
    esperado, _ = estimar_mu_michaud(
        r, n_replicas=8, rf=0.0, peso_maximo=1.0, devolver_diagnostico=True,
        rng=np.random.default_rng(7), frecuencia=252,
    )
    res = modelos.MichaudOptimizer(n_replicas=8, frequency=252, seed=7).optimize(
        mu, cov, retornos_historicos=r
    )
    assert _dif(list(res["weights"].values()), esperado) < 1e-12


@pytest.mark.parametrize("semilla", SEMILLAS)
def test_efficient_risk_markowitz_igual_a_volatilidad_objetivo_canonica(semilla):
    r = _retornos(semilla)
    mu, cov = r.mean() * 252, r.cov() * 252
    esperado, _ = optimizar_volatilidad_objetivo(mu.values, cov.values, 0.14, peso_maximo=0.5)
    res = modelos.MarkowitzOptimizer("efficient_risk", target_volatility=0.14).optimize(
        mu, cov, {"max_weight": 0.5}
    )
    assert _dif(list(res["weights"].values()), esperado) < 1e-12


@pytest.mark.parametrize("semilla", SEMILLAS)
@pytest.mark.parametrize("peso_maximo", [1.0, 0.3])
def test_paridad_riesgo_funcion_igual_a_clase(semilla, peso_maximo):
    r = _retornos(semilla)
    cov = r.cov() * 252
    restricciones = {"max_weight": peso_maximo}
    desde_funcion = legado.optimizar_risk_parity(cov, restricciones)
    desde_clase = modelos.RiskParityOptimizer().optimize(r.mean() * 252, cov, restricciones)["weights"]
    assert _dif(desde_funcion, list(desde_clase.values())) < 1e-10


@pytest.mark.parametrize("semilla", SEMILLAS)
def test_cvar_diario_del_optimizador_igual_a_calcular_cvar(semilla):
    r = _retornos(semilla)
    pesos = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    diario = modelos.PMPTMinCVaROptimizer(0.95)._calcular_cvar(pesos, r)
    assert diario == pytest.approx(legado.calcular_cvar(r, pesos, 0.95)["tail_mean"] / 252, abs=1e-12)


# ---------------------------------------------------------------------------------------
# Diferencias medidas: NO coinciden; la cota es la diferencia máxima de pesos observada (con margen)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("semilla", SEMILLAS)
@pytest.mark.parametrize("peso_maximo", [1.0, 0.3])
def test_max_sharpe_scipy_difiere_del_qp_exacto_por_tolerancia_del_solver(semilla, peso_maximo):
    """SLSQP (retorno/vol sin rf) frente al QP exacto: medido hasta 1,4e-4 en pesos."""
    r = _retornos(semilla)
    mu, cov = r.mean() * 252, r.cov() * 252
    qp, _, _ = optimizar_max_sharpe(mu.values, cov.values, 0.0, peso_maximo)
    scipy_ = legado.optimizar_max_sharpe_scipy(r, {"max_weight": peso_maximo})
    assert _dif(scipy_, qp) < 3e-4


@pytest.mark.parametrize("semilla", SEMILLAS)
@pytest.mark.parametrize("peso_maximo", [1.0, 0.3])
def test_min_volatilidad_scipy_difiere_de_min_varianza_qp(semilla, peso_maximo):
    """SLSQP frente a la GMV del QP: medido hasta 6,5e-4 en pesos."""
    r = _retornos(semilla)
    gmv = optimizar_min_varianza((r.cov() * 252).values, peso_maximo)
    scipy_ = legado.optimizar_min_volatilidad(r, {"max_weight": peso_maximo})
    assert _dif(scipy_, gmv) < 1.5e-3


@pytest.mark.parametrize("semilla", SEMILLAS)
@pytest.mark.parametrize("objetivo", [0.12, 0.15])
def test_volatilidad_objetivo_scipy_difiere_de_la_conica(semilla, objetivo):
    """SLSQP (sin tope de peso, sin GMV de respaldo) frente al SOCP: medido hasta 5,3e-4."""
    r = _retornos(semilla)
    mu, cov = r.mean() * 252, r.cov() * 252
    conica, _ = optimizar_volatilidad_objetivo(mu.values, cov.values, objetivo)
    scipy_ = legado.optimizar_volatilidad_objetivo(r, objetivo)["weights"]
    assert _dif(list(scipy_.values()), conica) < 1.5e-3


@pytest.mark.parametrize("semilla", SEMILLAS)
@pytest.mark.parametrize("rf", [0.0, 0.02])
def test_markowitz_pypfopt_difiere_del_qp_por_clean_weights(semilla, rf):
    """pypfopt.max_sharpe + `clean_weights` (redondeo a 5 decimales) frente al QP: medido hasta 5e-6."""
    r = _retornos(semilla)
    mu, cov = r.mean() * 252, r.cov() * 252
    qp, _, _ = optimizar_max_sharpe(mu.values, cov.values, rf)
    markowitz = modelos.MarkowitzOptimizer("max_sharpe", rf).optimize(mu, cov)["weights"]
    assert _dif(list(markowitz.values()), qp) < 5e-5


def test_semicovarianza_pypfopt_no_equivale_a_la_de_ceros():
    """
    `SemiCovarianceModel` (pypfopt: promedia productos de excesos negativos, n) y
    `calcular_semivarianza` (pone a 0 los retornos sobre el umbral y usa cov(), con centrado y n-1)
    son estimadores distintos: diferencia relativa medida 38 % con estos datos.
    """
    r = _retornos(1)
    pypfopt = modelos.SemiCovarianceModel(0.0, 252).calculate(r)
    ceros = legado.calcular_semivarianza(r, 0.0)
    relativa = _dif(pypfopt, ceros) / float(np.abs(ceros.values).max())
    assert 0.1 < relativa < 0.7
