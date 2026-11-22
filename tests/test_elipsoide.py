"""
Tests de la optimización robusta con incertidumbre elipsoidal y de caja en µ (v0.5.0).

Responsabilidad: verificar `estimacion_robusta.elipsoide` (casos límite κ=0 y κ grande, peor caso
<= nominal, monotonía en κ, restricciones, validaciones), los optimizadores Strategy
(`RobustoElipsoideOptimizer`, `RobustoCajaOptimizer`), el preset `robusta_elipsoide` y las
estrategias `robusta_elipsoide` / `robusta_caja` de extremo a extremo con precios sintéticos.

Se relaciona con: src/ypyquant/optimization/estimacion_robusta/elipsoide.py,
optimization/portfolio_models.py, optimization/presets/robusta.py,
estrategias/incluidas.py y tests/test_estrategias.py (cargador sintético).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.stats import chi2

from ypyquant.estrategias import registro
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, ejecutar_experimento, expandir_rejilla
from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.optimization import estimacion_robusta as er
from ypyquant.optimization.optimizador_portafolio import crear_modelo_personalizado
from ypyquant.optimization.portfolio_models import (
    OptimizerType,
    RobustoCajaOptimizer,
    RobustoElipsoideOptimizer,
)
from ypyquant.optimization.presets import listar_presets, obtener_preset

T_OBS = 120


def _mu_sigma(n: int = 6, semilla: int = 3):
    """µ y Σ anualizados sintéticos (Σ definida positiva, volatilidades 12-35 %)."""
    rng = np.random.default_rng(semilla)
    a = rng.normal(size=(n, n))
    corr = a @ a.T + n * np.eye(n)
    d = 1.0 / np.sqrt(np.diag(corr))
    corr = corr * np.outer(d, d)
    vol = np.linspace(0.12, 0.35, n)
    return rng.uniform(0.02, 0.15, size=n), corr * np.outer(vol, vol)


def _media_varianza_scipy(mu, sigma, aversion, peso_maximo=1.0):
    """Referencia independiente (SLSQP): max µ'w - (λ/2)w'Σw long-only con suma 1."""
    n = len(mu)
    res = minimize(
        lambda w: -(mu @ w) + 0.5 * aversion * w @ sigma @ w, np.full(n, 1 / n), method="SLSQP",
        bounds=[(0, peso_maximo)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    return res.x


# ---- construcción de la incertidumbre -----------------------------------------


def test_matriz_incertidumbre_sigma_diagonal_y_frecuencia():
    _, sigma = _mu_sigma()
    np.testing.assert_allclose(er.matriz_incertidumbre(sigma, 100), sigma / 100)
    np.testing.assert_allclose(er.matriz_incertidumbre(sigma, 100, "diagonal"), np.diag(np.diag(sigma)) / 100)
    np.testing.assert_allclose(er.matriz_incertidumbre(sigma, 100, frecuencia=252), sigma * 2.52)
    with pytest.raises(ValueError):
        er.matriz_incertidumbre(sigma, 0)
    with pytest.raises(ValueError):
        er.matriz_incertidumbre(sigma, 10, "otra")


def test_kappa_confianza_es_raiz_del_cuantil_chi2():
    assert er.kappa_confianza(0.9, 5) == pytest.approx(np.sqrt(chi2.ppf(0.9, 5)))
    assert er.kappa_confianza(0.0, 5) == 0.0
    # Con N grande κ ≈ √N: crece mucho
    assert er.kappa_confianza(0.9, 100) > 10 > er.kappa_confianza(0.9, 3)
    with pytest.raises(ValueError):
        er.kappa_confianza(1.0, 5)


def test_delta_caja_escala_con_z_y_sqrt_t():
    _, sigma = _mu_sigma()
    delta = er.delta_caja(sigma, 100, 2.0)
    np.testing.assert_allclose(delta, 2.0 * np.sqrt(np.diag(sigma) / 100))
    assert er.z_confianza(0.95) == pytest.approx(1.959964, abs=1e-5)
    assert er.z_confianza(0.0) == 0.0


# ---- elipsoide: casos límite ---------------------------------------------------


def test_kappa_cero_reproduce_media_varianza():
    mu, sigma = _mu_sigma()
    theta = er.matriz_incertidumbre(sigma, T_OBS)
    pesos, estado, _ = er.optimizar_robusto_elipsoide(mu, sigma, theta, 0.0, aversion=4.0)
    assert estado == "robusto_elipsoide"
    np.testing.assert_allclose(pesos, _media_varianza_scipy(mu, sigma, 4.0), atol=1e-4)


def test_kappa_cero_con_volatilidad_objetivo_coincide_con_optimizar_volatilidad_objetivo():
    mu, sigma = _mu_sigma()
    theta = er.matriz_incertidumbre(sigma, T_OBS)
    objetivo = 0.20
    pesos, estado, _ = er.optimizar_robusto_elipsoide(
        mu, sigma, theta, 0.0, aversion=0.0, volatilidad_objetivo=objetivo, peso_maximo=0.5
    )
    esperados, estado_ref = er.optimizar_volatilidad_objetivo(mu, sigma, objetivo, peso_maximo=0.5)
    assert estado == estado_ref
    np.testing.assert_allclose(pesos, esperados, atol=1e-4)


def test_kappa_grande_tiende_a_la_gmv_con_theta_proporcional_a_sigma():
    mu, sigma = _mu_sigma()
    theta = er.matriz_incertidumbre(sigma, T_OBS)
    pesos, _, _ = er.optimizar_robusto_elipsoide(mu, sigma, theta, 1e4)
    np.testing.assert_allclose(pesos, er.optimizar_min_varianza(sigma), atol=2e-3)


def test_theta_diagonal_penaliza_los_activos_inciertos():
    mu, sigma = _mu_sigma()
    theta = np.diag(np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 5e-2]))  # el último µ̂ es muy incierto
    base, _, _ = er.optimizar_robusto_elipsoide(mu, sigma, theta, 0.0, aversion=2.0)
    robusta, _, _ = er.optimizar_robusto_elipsoide(mu, sigma, theta, 3.0, aversion=2.0)
    assert robusta[-1] <= base[-1] + 1e-9


# ---- elipsoide: propiedades ----------------------------------------------------


def test_peor_caso_menor_o_igual_que_nominal_y_diagnostico_consistente():
    mu, sigma = _mu_sigma()
    theta = er.matriz_incertidumbre(sigma, T_OBS, frecuencia=252)
    kappa = er.kappa_confianza(0.9, len(mu))
    pesos, _, diag = er.optimizar_robusto_elipsoide(mu, sigma, theta, kappa, aversion=2.0)
    assert diag["rentabilidad_peor_caso"] <= diag["rentabilidad_nominal"] + 1e-12
    assert diag["rentabilidad_nominal"] == pytest.approx(float(mu @ pesos))
    assert diag["volatilidad"] == pytest.approx(float(np.sqrt(pesos @ sigma @ pesos)))
    assert diag["penalizacion"] == pytest.approx(kappa * float(np.sqrt(pesos @ theta @ pesos)))
    assert diag["rentabilidad_nominal"] - diag["rentabilidad_peor_caso"] == pytest.approx(diag["penalizacion"])


def test_monotonia_en_kappa_con_volatilidad_objetivo():
    """Al subir κ no aumentan ni la penalización ‖Θ^½w‖ ni la rentabilidad nominal (aversión 0)."""
    mu, sigma = _mu_sigma()
    theta = er.matriz_incertidumbre(sigma, T_OBS, frecuencia=252)
    previa = None
    for kappa in (0.0, 0.5, 1.0, 2.0, 5.0, 20.0):
        pesos, _, diag = er.optimizar_robusto_elipsoide(
            mu, sigma, theta, kappa, aversion=0.0, volatilidad_objetivo=0.30
        )
        riesgo_estimacion = float(np.sqrt(pesos @ theta @ pesos))
        nominal = diag["rentabilidad_nominal"]
        if previa is not None:
            assert riesgo_estimacion <= previa[0] + 1e-6
            assert nominal <= previa[1] + 1e-6
        previa = (riesgo_estimacion, nominal)


def test_restricciones_suma_uno_long_only_tope_y_volatilidad():
    mu, sigma = _mu_sigma()
    theta = er.matriz_incertidumbre(sigma, T_OBS, frecuencia=252)
    objetivo = 0.22
    pesos, estado, diag = er.optimizar_robusto_elipsoide(
        mu, sigma, theta, 1.0, aversion=1.0, volatilidad_objetivo=objetivo, peso_maximo=0.35
    )
    assert pesos.sum() == pytest.approx(1.0, abs=1e-8)
    assert pesos.min() >= -1e-9 and pesos.max() <= 0.35 + 1e-6
    assert diag["volatilidad"] <= objetivo * (1 + 1e-4)
    assert estado in ("volatilidad_objetivo", "volatilidad_objetivo_holgada")


def test_volatilidad_inalcanzable_devuelve_gmv():
    mu, sigma = _mu_sigma()
    theta = er.matriz_incertidumbre(sigma, T_OBS)
    pesos, estado, _ = er.optimizar_robusto_elipsoide(mu, sigma, theta, 1.0, volatilidad_objetivo=0.01)
    assert estado == "gmv_volatilidad_inalcanzable"
    np.testing.assert_allclose(pesos, er.optimizar_min_varianza(sigma), atol=1e-8)


def test_theta_semidefinida_singular_no_rompe():
    mu, sigma = _mu_sigma()
    theta = np.ones((6, 6)) * 1e-3  # rango 1
    pesos, estado, _ = er.optimizar_robusto_elipsoide(mu, sigma, theta, 2.0, aversion=1.0)
    assert estado == "robusto_elipsoide" and pesos.sum() == pytest.approx(1.0, abs=1e-8)


def test_validacion_de_entradas():
    mu, sigma = _mu_sigma()
    theta = er.matriz_incertidumbre(sigma, T_OBS)
    with pytest.raises(ValueError, match="kappa"):
        er.optimizar_robusto_elipsoide(mu, sigma, theta, -1.0)
    with pytest.raises(ValueError, match="simétrica"):
        er.optimizar_robusto_elipsoide(mu, np.triu(sigma), theta, 1.0)
    with pytest.raises(ValueError, match="Dimensiones"):
        er.optimizar_robusto_elipsoide(mu[:-1], sigma, theta, 1.0)
    with pytest.raises(ValueError, match="theta"):
        er.optimizar_robusto_elipsoide(mu, sigma, theta[:-1, :-1], 1.0)
    with pytest.raises(ValueError, match="aversion"):
        er.optimizar_robusto_elipsoide(mu, sigma, theta, 1.0, aversion=-1.0)
    with pytest.raises(ValueError, match="peso_maximo"):
        er.optimizar_robusto_elipsoide(mu, sigma, theta, 1.0, peso_maximo=0.1)
    with pytest.raises(ValueError, match="volatilidad_objetivo"):
        er.optimizar_robusto_elipsoide(mu, sigma, theta, 1.0, volatilidad_objetivo=-0.1)
    with pytest.raises(ValueError, match="delta"):
        er.optimizar_robusto_caja(mu, sigma, -np.ones(6))


# ---- caja ----------------------------------------------------------------------


def test_caja_con_delta_cero_es_media_varianza():
    mu, sigma = _mu_sigma()
    pesos, estado, diag = er.optimizar_robusto_caja(mu, sigma, np.zeros(6), aversion=4.0)
    assert estado == "robusto_caja"
    np.testing.assert_allclose(pesos, _media_varianza_scipy(mu, sigma, 4.0), atol=1e-4)
    assert diag["rentabilidad_peor_caso"] == pytest.approx(diag["rentabilidad_nominal"])


def test_caja_peor_caso_y_restricciones():
    mu, sigma = _mu_sigma()
    delta = er.delta_caja(sigma, T_OBS, 2.0, frecuencia=252)
    pesos, _, diag = er.optimizar_robusto_caja(
        mu, sigma, delta, aversion=1.0, volatilidad_objetivo=0.25, peso_maximo=0.4
    )
    assert diag["rentabilidad_peor_caso"] == pytest.approx(float((mu - delta) @ pesos))
    assert diag["rentabilidad_peor_caso"] <= diag["rentabilidad_nominal"]
    assert pesos.sum() == pytest.approx(1.0, abs=1e-8) and pesos.max() <= 0.4 + 1e-6
    assert diag["volatilidad"] <= 0.25 * (1 + 1e-4)


def test_caja_delta_escalar_y_monotonia_de_la_penalizacion():
    mu, sigma = _mu_sigma()
    previa = np.inf
    for escala in (0.0, 0.01, 0.05, 0.5):
        pesos, _, diag = er.optimizar_robusto_caja(mu, sigma, escala, aversion=0.0, volatilidad_objetivo=0.30)
        assert diag["penalizacion"] == pytest.approx(escala * pesos.sum())
        assert diag["rentabilidad_nominal"] <= previa + 1e-6 or previa == np.inf
        previa = diag["rentabilidad_nominal"]


# ---- Strategy pattern y preset -------------------------------------------------


def _retornos_diarios(n_dias: int = 300, n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    vol = np.linspace(0.008, 0.02, n)
    datos = rng.normal(np.linspace(0.0002, 0.0009, n), vol, size=(n_dias, n))
    return pd.DataFrame(datos, columns=[f"A{i}" for i in range(n)])


def test_optimizador_elipsoide_usa_longitud_de_la_ventana_como_t():
    retornos = _retornos_diarios()
    modelo = crear_modelo_personalizado(
        "m", "historical_mean", "ledoit_wolf", OptimizerType.ROBUSTO_ELIPSOIDE.value,
        optimizer_config={"confianza": 0.8, "volatilidad_objetivo": 0.15},
    )
    assert isinstance(modelo.optimizer, RobustoElipsoideOptimizer)
    resultado = modelo.optimize(retornos, constraints={"max_weight": 0.5})
    meta = resultado["optimizer_metadata"]
    assert meta["n_observaciones"] == len(retornos)
    assert meta["kappa"] == pytest.approx(er.kappa_confianza(0.8, 5))
    assert sum(resultado["weights"].values()) == pytest.approx(1.0, abs=1e-6)
    assert max(resultado["weights"].values()) <= 0.5 + 1e-6
    assert meta["diagnostico"]["rentabilidad_peor_caso"] <= meta["diagnostico"]["rentabilidad_nominal"]


def test_optimizador_kappa_directo_prevalece_y_requiere_historico():
    retornos = _retornos_diarios()
    modelo = crear_modelo_personalizado(
        "m", "historical_mean", "ledoit_wolf", "robusto_elipsoide", optimizer_config={"kappa": 1.5, "confianza": 0.99}
    )
    assert modelo.optimize(retornos)["optimizer_metadata"]["kappa"] == 1.5
    optimizador = RobustoElipsoideOptimizer()
    mu = pd.Series([0.1, 0.05], index=["A", "B"])
    sigma = pd.DataFrame(np.diag([0.04, 0.02]), index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError, match="retornos_historicos"):
        optimizador.optimize(mu, sigma)
    assert RobustoElipsoideOptimizer(n_observaciones=100).optimize(mu, sigma)["weights"]


def test_optimizador_caja_via_modelo_personalizado():
    modelo = crear_modelo_personalizado("m", "historical_mean", "ledoit_wolf", "robusto_caja", optimizer_config={"z": 1.0})
    assert isinstance(modelo.optimizer, RobustoCajaOptimizer)
    resultado = modelo.optimize(_retornos_diarios())
    assert sum(resultado["weights"].values()) == pytest.approx(1.0, abs=1e-6)
    assert resultado["optimizer_metadata"]["z"] == 1.0


def test_optimizador_avisa_si_la_volatilidad_es_inalcanzable():
    modelo = crear_modelo_personalizado(
        "m", "historical_mean", "ledoit_wolf", "robusto_elipsoide", optimizer_config={"volatilidad_objetivo": 0.001}
    )
    meta = modelo.optimize(_retornos_diarios())["optimizer_metadata"]
    assert meta["method"] == "gmv_volatilidad_inalcanzable" and meta["avisos"]


def test_preset_robusta_elipsoide_se_construye_y_optimiza():
    assert "robusta_elipsoide" in listar_presets()
    modelo = obtener_preset("robusta_elipsoide")
    assert isinstance(modelo.optimizer, RobustoElipsoideOptimizer)
    resultado = modelo.optimize(_retornos_diarios())
    assert sum(resultado["weights"].values()) == pytest.approx(1.0, abs=1e-6)
    assert all(w >= -1e-9 for w in resultado["weights"].values())


# ---- estrategias de extremo a extremo ------------------------------------------

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _config(**cambios) -> ConfiguracionExperimento:
    base = {"universo": TICKERS, "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}
    return ConfiguracionExperimento(**{**base, **cambios})


def test_estrategias_registradas_con_schema_para_la_ui():
    for nombre, extra in (("robusta_elipsoide", {"kappa", "tipo_theta"}), ("robusta_caja", {"z"})):
        propiedades = registro.descripcion_estrategia(nombre)["parametros_schema"]["properties"]
        comunes = {"estimador_mu", "confianza", "aversion", "volatilidad_objetivo", "peso_maximo", "ventana_dias"}
        assert (comunes | extra) <= set(propiedades)
        assert all(propiedades[c].get("description") for c in comunes | extra)


@pytest.mark.parametrize("estrategia", ["robusta_elipsoide", "robusta_caja"])
@pytest.mark.parametrize("estimador_mu", ["muestral", "bayes_stein", "bl_prior"])
def test_estrategia_robusta_ejecuta_de_extremo_a_extremo(estrategia, estimador_mu):
    resultado = ejecutar_experimento(
        _config(estrategia=estrategia, parametros={"estimador_mu": estimador_mu, "peso_maximo": 0.4}),
        FuenteSintetica(),
    )
    assert resultado.metricas["n_rebalanceos"] > 5 and not resultado.pesos.empty
    assert resultado.pesos.groupby("fecha")["peso"].sum().round(6).eq(1.0).all()
    assert resultado.pesos["peso"].min() >= -1e-9 and resultado.pesos["peso"].max() <= 0.4 + 1e-6


def test_estrategia_elipsoide_con_volatilidad_objetivo_y_rejilla():
    base = _config(estrategia="robusta_elipsoide", parametros={"volatilidad_objetivo": 0.10})
    configs = expandir_rejilla(base, {"kappa": [0.0, 1.0], "tipo_theta": ["sigma", "diagonal"]})
    assert len(configs) == 4
    resultados = [ejecutar_experimento(c, FuenteSintetica()) for c in configs]
    assert all(r.metricas["n_rebalanceos"] > 5 for r in resultados)
    # κ distinto → carteras distintas
    huellas = {tuple(r.pesos["peso"].round(5)) for r in resultados}
    assert len(huellas) > 1


def test_kappa_directo_prevalece_sobre_confianza_en_la_estrategia():
    comun = {"estrategia": "robusta_elipsoide"}
    a = ejecutar_experimento(_config(**comun, parametros={"kappa": 1.0, "confianza": 0.5}), FuenteSintetica())
    b = ejecutar_experimento(_config(**comun, parametros={"kappa": 1.0, "confianza": 0.99}), FuenteSintetica())
    np.testing.assert_allclose(a.pesos["peso"].to_numpy(), b.pesos["peso"].to_numpy(), atol=1e-6)
