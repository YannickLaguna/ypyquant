"""
test_estimacion_robusta.py — Tests del subpaquete ypyquant.optimization.estimacion_robusta.

Responsabilidad: verificar las propiedades críticas de los estimadores robustos del TFM (portadas de
`tests/test_estrategias_rejilla_v2.py` del TFM, con imports adaptados) y lo nuevo de v0.4.0:
- BL-prior colapsa a w_mkt, Woodbury = fórmula directa, BL-momentum (Ω, ventana 12-1), diagnóstico BL.
- Solver max-Sharpe (contingencia, QP vs SLSQP, penalización de turnover).
- Michaud (iid y bootstrap estacionario por bloques), Bayes-Stein (escala de φ, aviso T < N+2).
- Universo: anidamiento exacto y elegibilidad point-in-time SIN look-ahead.
- w_mkt: proxy por volumen y proveedor de capitalización inyectable.
- Motor de rejilla sobre datos sintéticos (turnover frente a deriva, coste, contingencia).
- Integración Strategy: `crear_modelo_personalizado`, `modelos_disponibles`, presets `tfm_*` y `pmpt_cvar`.
Datos sintéticos, sin BD ni red (la paridad con el TFM está en test_paridad_tfm.py, marcado `datos`).

Se relaciona con: src/ypyquant/optimization/estimacion_robusta/, portfolio_models.py,
optimizador_portafolio.py y presets/{tfm,pmpt}.py.
"""

import numpy as np
import pandas as pd
import pytest

from ypyquant.optimization import estimacion_robusta as er
from ypyquant.optimization.estimacion_robusta import max_sharpe_qp as qp
from ypyquant.optimization.optimizador_portafolio import crear_modelo_personalizado, modelos_disponibles
from ypyquant.optimization.portfolio_models import (
    BayesSteinReturnsModel,
    BLMomentumReturnsModel,
    BLPriorReturnsModel,
    LedoitWolfShrinkageModel,
    MaxSharpeQPOptimizer,
    MichaudOptimizer,
    PMPTMinCVaROptimizer,
)
from ypyquant.optimization.presets import listar_presets, obtener_preset

FREC = 12


def _sigma_spd_aleatoria(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _w_mkt_aleatorio(n: int, seed: int = 1) -> np.ndarray:
    w = np.random.default_rng(seed).random(n)
    return w / w.sum()


def _retornos_sinteticos(t: int, n: int, seed: int = 0, media: float = 0.008, vol: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 0.03, size=(t, 1))
    datos = media + factor + rng.normal(0, vol, size=(t, n))
    return pd.DataFrame(datos, columns=[f"A{i}" for i in range(n)])


# ============================================================================
# Σ Ledoit-Wolf
# ============================================================================


class TestSigma:
    @pytest.mark.parametrize("frecuencia", [12, 252])
    def test_coincide_con_ledoit_wolf_shrinkage_model(self, frecuencia):
        """Mismo resultado (bit a bit) que LedoitWolfShrinkageModel de portfolio_models para igual frecuencia."""
        retornos = _retornos_sinteticos(60, 8, seed=3)
        propia = er.estimar_sigma_ledoit_wolf(retornos, frecuencia)
        existente = LedoitWolfShrinkageModel(frequency=frecuencia).calculate(retornos)
        np.testing.assert_array_equal(propia.values, existente.values)


# ============================================================================
# BL-prior
# ============================================================================


class TestBLPrior:
    def test_identidad_algebraica_sigma_inv_pi_es_delta_w_mkt(self):
        n = 8
        sigma = _sigma_spd_aleatoria(n)
        w_mkt = _w_mkt_aleatorio(n)
        pi = er.estimar_mu_bl_prior(pd.DataFrame(sigma), w_mkt, delta=2.5)
        np.testing.assert_allclose(np.linalg.solve(sigma, pi), 2.5 * w_mkt, atol=1e-8)

    def test_qp_recupera_w_mkt_exacto(self):
        """El QP real alimentado con µ=Π reproduce w_mkt (justifica el atajo w=w_mkt del motor)."""
        n = 6
        sigma = _sigma_spd_aleatoria(n)
        w_mkt = _w_mkt_aleatorio(n)
        pi = er.estimar_mu_bl_prior(pd.DataFrame(sigma), w_mkt, delta=2.5)
        pesos, degenerate, _ = er.optimizar_max_sharpe(pi, sigma, rf=0.0)
        assert not degenerate
        np.testing.assert_allclose(pesos, w_mkt, atol=1e-4)

    @pytest.mark.parametrize("n", [20, 150])
    @pytest.mark.parametrize("delta", [2.5, 5.0])
    def test_qp_recupera_w_mkt_para_n_y_delta(self, n, delta):
        sigma = _sigma_spd_aleatoria(n, seed=n)
        w_mkt = _w_mkt_aleatorio(n, seed=n + 1)
        pi = er.estimar_mu_bl_prior(pd.DataFrame(sigma), w_mkt, delta=delta)
        pesos, degenerate, status = er.optimizar_max_sharpe(pi, sigma, rf=0.0)
        assert not degenerate, status
        assert np.max(np.abs(pesos - w_mkt)) < 1e-6


# ============================================================================
# BL-momentum
# ============================================================================


class TestBLMomentum:
    def test_omega_es_varianza_sobre_n_obs(self):
        ventana = pd.DataFrame(np.random.default_rng(2).normal(0.01, 0.05, size=(11, 4)))
        omega = er.calcular_omega_momentum(ventana)
        np.testing.assert_allclose(omega, np.maximum(ventana.var().values * 12, 1e-10) / 11)
        assert np.all(omega < ventana.var().values * 12 / 5)

    def test_woodbury_igual_a_formula_directa(self):
        """µ_bl y M de Woodbury coinciden con la forma matricial directa He-Litterman (P=I)."""
        n = 6
        sigma = _sigma_spd_aleatoria(n, seed=40) / 500
        w_mkt = _w_mkt_aleatorio(n, seed=41)
        ventana = pd.DataFrame(np.random.default_rng(42).normal(0.01, 0.05, size=(11, n)))
        tau, delta = 0.05, 2.5

        mu_bl, m = er.estimar_mu_bl_momentum(ventana, pd.DataFrame(sigma), w_mkt, delta=delta, tau=tau)

        pi = delta * sigma @ w_mkt
        q = ventana.mean().values * 12
        omega = np.diag(er.calcular_omega_momentum(ventana))
        inv_tau_sigma = np.linalg.inv(tau * sigma)
        precision = inv_tau_sigma + np.linalg.inv(omega)          # (τΣ)⁻¹ + PᵀΩ⁻¹P
        mu_directo = np.linalg.solve(precision, inv_tau_sigma @ pi + np.linalg.inv(omega) @ q)
        m_directa = np.linalg.inv(precision)
        np.testing.assert_allclose(mu_bl, mu_directo, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(m, m_directa, rtol=1e-8, atol=1e-12)

    def test_senal_usa_ventana_t12_t2(self):
        valores = np.tile(np.arange(12, 23).reshape(-1, 1), (1, 3)).astype(float)
        sigma = pd.DataFrame(_sigma_spd_aleatoria(3, seed=3) / 1000)
        mu_bl, _ = er.estimar_mu_bl_momentum(pd.DataFrame(valores), sigma, np.full(3, 1 / 3), delta=2.5, tau=1e6)
        np.testing.assert_allclose(mu_bl, np.arange(12, 23).mean() * 12, rtol=0.05)

    def test_omega_infinito_recupera_prior(self):
        n, tau = 5, 0.05
        sigma = _sigma_spd_aleatoria(n, seed=4)
        w_mkt = _w_mkt_aleatorio(n, seed=5)
        ventana = pd.DataFrame(np.random.default_rng(6).normal(0, 1e6, size=(11, n)))
        mu_bl, m = er.estimar_mu_bl_momentum(ventana, pd.DataFrame(sigma), w_mkt, delta=2.5, tau=tau)
        np.testing.assert_allclose(mu_bl, er.estimar_mu_bl_prior(pd.DataFrame(sigma), w_mkt), rtol=1e-3)
        np.testing.assert_allclose(m, tau * sigma, rtol=1e-3)

    def test_ventana_vacia_devuelve_prior(self):
        sigma = pd.DataFrame(_sigma_spd_aleatoria(3, seed=7))
        mu, m = er.estimar_mu_bl_momentum(pd.DataFrame(columns=range(3)), sigma, np.full(3, 1 / 3), tau=0.05)
        np.testing.assert_allclose(mu, er.estimar_mu_bl_prior(sigma, np.full(3, 1 / 3)))
        np.testing.assert_allclose(m, 0.05 * sigma.values)


class TestDiagnosticoBL:
    def test_omega_infinito_conf_conjunta_cero(self):
        n = 6
        ventana = pd.DataFrame(np.random.default_rng(22).normal(0, 1e6, size=(11, n)))
        diag = er.calcular_diagnostico_bl(ventana, pd.DataFrame(_sigma_spd_aleatoria(n, seed=20)),
                                          _w_mkt_aleatorio(n, seed=21))
        assert abs(diag["conf_conjunta"]) < 0.05

    def test_omega_diminuto_conf_conjunta_uno(self):
        n = 6
        rng = np.random.default_rng(25)
        base = rng.normal(0.05, 0.02, size=n)
        ventana = pd.DataFrame(np.tile(base, (11, 1)) + rng.normal(0, 1e-9, size=(11, n)))
        diag = er.calcular_diagnostico_bl(ventana, pd.DataFrame(_sigma_spd_aleatoria(n, seed=23) / 100),
                                          _w_mkt_aleatorio(n, seed=24))
        assert diag["conf_conjunta"] == pytest.approx(1.0, abs=0.05)

    def test_tau_cero_y_ventana_vacia(self):
        n = 5
        sigma = pd.DataFrame(_sigma_spd_aleatoria(n, seed=26))
        ventana = pd.DataFrame(np.random.default_rng(28).normal(0.01, 0.05, size=(11, n)))
        assert er.calcular_diagnostico_bl(ventana, sigma, _w_mkt_aleatorio(n), tau=0.0)["conf_aislada"] == 0.0
        vacio = er.calcular_diagnostico_bl(pd.DataFrame(columns=range(n)), sigma, np.full(n, 1 / n))
        assert all(np.isnan(v) for v in vacio.values())


# ============================================================================
# Solver
# ============================================================================


class TestSolver:
    def test_degenerate_cuando_todos_mu_no_positivos(self):
        sigma = _sigma_spd_aleatoria(5, seed=7)
        mu = -np.abs(np.random.default_rng(8).normal(size=5))
        pesos, degenerate, status = er.optimizar_max_sharpe(mu, sigma)
        assert degenerate is True
        assert status == "gmv_contingencia"
        np.testing.assert_allclose(pesos.sum(), 1.0, atol=1e-6)
        assert np.all(pesos >= -1e-9)

    def test_qp_pesos_validos_y_tope_de_peso(self):
        sigma = _sigma_spd_aleatoria(10, seed=9)
        mu = np.random.default_rng(10).uniform(0.01, 0.15, size=10)
        pesos, degenerate, status = er.optimizar_max_sharpe(mu, sigma)
        assert status in ("qp", "slsqp", "pypfopt") and not degenerate
        np.testing.assert_allclose(pesos.sum(), 1.0, atol=1e-6)
        topados, _, _ = er.optimizar_max_sharpe(mu, sigma, peso_maximo=0.2)
        assert topados.max() <= 0.2 + 1e-6

    def test_qp_coincide_con_slsqp(self):
        from scipy.optimize import minimize

        n = 15
        sigma = _sigma_spd_aleatoria(n, seed=11)
        mu = np.random.default_rng(12).uniform(0.01, 0.15, size=n)
        w_qp = qp._obtener_qp(n, 1.0).resolver(mu, sigma)

        def neg_sharpe(w):
            exc, vol = float(w @ mu), float(np.sqrt(w @ sigma @ w))
            return 1e6 if exc <= 0 or vol < 1e-12 else -exc / vol

        res = minimize(neg_sharpe, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-9, "maxiter": 500})
        w_slsqp = np.maximum(res.x, 0)
        assert np.abs(w_qp - w_slsqp / w_slsqp.sum()).sum() < 0.05

    def test_penalizacion_cero_es_el_qp_exacto(self):
        sigma = _sigma_spd_aleatoria(8, seed=13)
        mu = np.random.default_rng(14).uniform(0.02, 0.12, size=8)
        base = er.optimizar_max_sharpe(mu, sigma)
        con_cero = er.optimizar_max_sharpe(mu, sigma, pesos_previos=np.full(8, 1 / 8), penalizacion_turnover=0.0)
        np.testing.assert_allclose(base[0], con_cero[0], atol=1e-9)  # warm_start de OSQP: ruido ~1e-14
        assert base[2] == con_cero[2]

    def test_penalizacion_reduce_turnover_y_converge_a_pesos_previos(self):
        n = 8
        sigma = _sigma_spd_aleatoria(n, seed=13)
        mu = np.random.default_rng(14).uniform(0.02, 0.12, size=n)
        previos = np.full(n, 1 / n)
        tangente, _, _ = er.optimizar_max_sharpe(mu, sigma)
        pesos_pequena, _, estado = er.optimizar_max_sharpe(mu, sigma, pesos_previos=previos, penalizacion_turnover=1e-6)
        pesos_media, _, _ = er.optimizar_max_sharpe(mu, sigma, pesos_previos=previos, penalizacion_turnover=0.05)
        pesos_grande, _, _ = er.optimizar_max_sharpe(mu, sigma, pesos_previos=previos, penalizacion_turnover=1e3)
        assert estado == "qp_turnover"
        # λ→0 recupera la tangente (el δ elegido hace equivalentes los dos problemas)
        np.testing.assert_allclose(pesos_pequena, tangente, atol=5e-3)
        # más penalización → más cerca de los pesos previos
        dist = [np.abs(w - previos).sum() for w in (tangente, pesos_media, pesos_grande)]
        assert dist[0] >= dist[1] >= dist[2]
        assert dist[2] < 1e-3

    def test_penalizacion_longitud_incorrecta_lanza(self):
        sigma = _sigma_spd_aleatoria(4, seed=15)
        with pytest.raises(ValueError):
            er.optimizar_max_sharpe(np.full(4, 0.05), sigma, pesos_previos=np.ones(3), penalizacion_turnover=0.1)


# ============================================================================
# Bootstrap por bloques y Michaud
# ============================================================================


class TestBootstrapYMichaud:
    def test_indices_en_rango_longitud_y_reproducibilidad(self):
        for metodo in ("iid", "bloques"):
            a = er.indices_bootstrap(40, metodo, 5.0, np.random.default_rng(1))
            b = er.indices_bootstrap(40, metodo, 5.0, np.random.default_rng(1))
            assert len(a) == 40 and a.min() >= 0 and a.max() < 40
            np.testing.assert_array_equal(a, b)

    def test_bloques_conservan_dependencia_serial_iid_no(self):
        """Fracción de pasos 'i → i+1' en la remuestra: ≈ 1 - 1/L en bloques, ≈ 1/n en iid."""
        n, longitud = 4000, 10.0
        idx_bloques = er.indices_bootstrap(n, "bloques", longitud, np.random.default_rng(7))
        idx_iid = er.indices_bootstrap(n, "iid", rng=np.random.default_rng(7))
        consecutivos_b = np.mean((np.diff(idx_bloques) % n) == 1)
        consecutivos_i = np.mean((np.diff(idx_iid) % n) == 1)
        assert consecutivos_b == pytest.approx(1 - 1 / longitud, abs=0.03)
        assert consecutivos_i < 0.01

    def test_longitud_uno_equivale_a_iid_en_estructura(self):
        idx = er.indices_bootstrap(3000, "bloques", 1.0, np.random.default_rng(3))
        assert np.mean((np.diff(idx) % 3000) == 1) < 0.01

    def test_metodo_invalido_y_longitud_invalida(self):
        with pytest.raises(ValueError):
            er.indices_bootstrap(10, "otro")
        with pytest.raises(ValueError):
            er.indices_bootstrap(10, "bloques", 0.5)

    @pytest.mark.parametrize("metodo", ["iid", "bloques"])
    def test_michaud_pesos_validos(self, metodo):
        retornos = _retornos_sinteticos(24, 6, seed=13)
        pesos = er.estimar_mu_michaud(retornos, n_replicas=5, metodo_bootstrap=metodo,
                                      rng=np.random.default_rng(0))
        assert pesos.shape == (6,)
        np.testing.assert_allclose(pesos.sum(), 1.0, atol=1e-6)
        assert np.all(pesos >= -1e-9)

    def test_michaud_reproducible_con_rng_y_bloques_difieren_de_iid(self):
        retornos = _retornos_sinteticos(36, 5, seed=21)
        a = er.estimar_mu_michaud(retornos, n_replicas=8, rng=np.random.default_rng(5))
        b = er.estimar_mu_michaud(retornos, n_replicas=8, rng=np.random.default_rng(5))
        c = er.estimar_mu_michaud(retornos, n_replicas=8, metodo_bootstrap="bloques", rng=np.random.default_rng(5))
        np.testing.assert_array_equal(a, b)
        assert not np.allclose(a, c)

    def test_michaud_converge_al_aumentar_replicas(self):
        """La dispersión entre corridas con distinta semilla baja al subir R (el promedio converge)."""
        retornos = _retornos_sinteticos(36, 5, seed=22)

        def dispersion(r):
            pesos = np.array([er.estimar_mu_michaud(retornos, n_replicas=r, rng=np.random.default_rng(s))
                              for s in range(6)])
            return pesos.std(axis=0).mean()

        assert dispersion(40) < dispersion(4)

    def test_michaud_diagnostico_devuelve_fraccion(self):
        retornos = _retornos_sinteticos(24, 4, seed=1, media=-0.05)  # medias negativas → contingencia frecuente
        pesos, frac = er.estimar_mu_michaud(retornos, n_replicas=6, devolver_diagnostico=True,
                                            rng=np.random.default_rng(1))
        assert 0.0 <= frac <= 1.0
        np.testing.assert_allclose(pesos.sum(), 1.0, atol=1e-6)


# ============================================================================
# Bayes-Stein
# ============================================================================


class TestBayesStein:
    @staticmethod
    def _phi_manual(retornos, sigma_np, n_anios):
        n = retornos.shape[1]
        mu_hat = retornos.mean().values * 12
        unos = np.ones(n)
        w = np.linalg.solve(sigma_np, unos)
        w = w / (unos @ w)
        dif = mu_hat - float(mu_hat @ w) * unos
        return float(np.clip((n + 2) / ((n + 2) + n_anios * (dif @ np.linalg.solve(sigma_np, dif))), 0, 1))

    def test_phi_usa_t_en_anios(self):
        retornos = _retornos_sinteticos(60, 15, seed=7, media=0.01, vol=0.04)
        sigma = er.estimar_sigma_ledoit_wolf(retornos)
        mu_bs = er.estimar_mu_bayes_stein(retornos, sigma)
        mu_hat = retornos.mean().values * 12
        unos = np.ones(15)
        w = np.linalg.solve(sigma.values, unos)
        mu_g = float(mu_hat @ (w / (unos @ w)))
        phi = self._phi_manual(retornos, sigma.values, 60 / 12)
        np.testing.assert_allclose(mu_bs, (1 - phi) * mu_hat + phi * mu_g * unos, rtol=1e-8)
        phi_bug = self._phi_manual(retornos, sigma.values, 60)
        assert not np.allclose(mu_bs, (1 - phi_bug) * mu_hat + phi_bug * mu_g * unos, rtol=1e-2)

    def test_mu_bs_entre_mu_hat_y_mu_gmv(self):
        retornos = _retornos_sinteticos(60, 10, seed=3)
        sigma = er.estimar_sigma_ledoit_wolf(retornos)
        mu_hat = retornos.mean().values * 12
        mu_bs = er.estimar_mu_bayes_stein(retornos, sigma)
        w = np.linalg.solve(sigma.values, np.ones(10))
        mu_g = float(mu_hat @ (w / w.sum()))
        assert np.all(mu_bs >= np.minimum(mu_hat, mu_g) - 1e-8) and np.all(mu_bs <= np.maximum(mu_hat, mu_g) + 1e-8)

    def test_mas_shrinkage_con_menos_observaciones(self):
        rng = np.random.default_rng(11)
        mu_true = rng.normal(0.01, 0.02, 12)
        cov_true = np.eye(12) * 0.002 + 0.001
        distancias = []
        for t_meses in (24, 240):
            muestra = pd.DataFrame(rng.multivariate_normal(mu_true, cov_true, size=t_meses))
            sigma = er.estimar_sigma_ledoit_wolf(muestra)
            mu_hat = muestra.mean().values * 12
            distancias.append(np.linalg.norm(er.estimar_mu_bayes_stein(muestra, sigma) - mu_hat) / np.linalg.norm(mu_hat))
        assert distancias[0] > distancias[1]

    def test_t_menor_n_mas_2_avisa_lanza_o_ignora(self):
        retornos = _retornos_sinteticos(12, 20, seed=2)  # T=12 < N+2=22
        sigma = er.estimar_sigma_ledoit_wolf(retornos)
        with pytest.warns(UserWarning, match="N\\+2"):
            er.estimar_mu_bayes_stein(retornos, sigma)
        with pytest.raises(ValueError, match="N\\+2"):
            er.estimar_mu_bayes_stein(retornos, sigma, si_t_menor_n2="error")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mu = er.estimar_mu_bayes_stein(retornos, sigma, si_t_menor_n2="ignorar")
        assert np.all(np.isfinite(mu))

    def test_sigma_singular_devuelve_mu_muestral(self):
        retornos = _retornos_sinteticos(30, 3, seed=5)
        singular = pd.DataFrame(np.ones((3, 3)))
        np.testing.assert_allclose(
            er.estimar_mu_bayes_stein(retornos, singular, si_t_menor_n2="ignorar"), retornos.mean().values * 12
        )


# ============================================================================
# w_mkt
# ============================================================================


class TestWMercado:
    def test_proxy_proporcional_a_dolar_volumen(self):
        fechas = pd.date_range("2019-01-31", periods=12, freq="ME")
        precios = pd.DataFrame(10.0, index=fechas, columns=["A", "B", "C"])
        volumen = pd.DataFrame({"A": [100] * 12, "B": [200] * 12, "C": [700] * 12}, index=fechas)
        w = er.calcular_w_mkt_volumen(precios, volumen, ["A", "B", "C"], fechas[-1], meses=12)
        np.testing.assert_allclose(w, [0.1, 0.2, 0.7], atol=1e-8)

    def test_proxy_no_usa_datos_posteriores_a_fecha_fin(self):
        fechas = pd.date_range("2019-01-31", periods=14, freq="ME")
        precios = pd.DataFrame(10.0, index=fechas, columns=["A", "B"])
        volumen = pd.DataFrame({"A": [100] * 14, "B": [100] * 14}, index=fechas)
        base = er.calcular_w_mkt_volumen(precios, volumen, ["A", "B"], fechas[11])
        volumen.loc[fechas[12]:, "B"] = 999999
        np.testing.assert_allclose(base, er.calcular_w_mkt_volumen(precios, volumen, ["A", "B"], fechas[11]))

    def test_proxy_sin_volumen_es_equiponderado(self):
        fechas = pd.date_range("2019-01-31", periods=12, freq="ME")
        precios = pd.DataFrame(10.0, index=fechas, columns=["A", "B"])
        w = er.calcular_w_mkt_volumen(precios, pd.DataFrame(0.0, index=fechas, columns=["A", "B"]), ["A", "B"], fechas[-1])
        np.testing.assert_allclose(w, [0.5, 0.5])

    def test_pesos_mercado_prefiere_proveedor_luego_proxy_luego_equiponderado(self):
        fechas = pd.date_range("2019-01-31", periods=12, freq="ME")
        activos = ["A", "B"]
        precios = pd.DataFrame(10.0, index=fechas, columns=activos)
        volumen = pd.DataFrame({"A": [100] * 12, "B": [300] * 12}, index=fechas)

        proveedor = er.CapitalizacionEstatica({"A": 3.0, "B": 1.0})
        assert isinstance(proveedor, er.ProveedorCapitalizacion)
        w, fuente = er.pesos_mercado(activos, fechas[-1], proveedor, precios, volumen)
        assert fuente == "capitalizacion"
        np.testing.assert_allclose(w, [0.75, 0.25])

        w, fuente = er.pesos_mercado(activos, fechas[-1], None, precios, volumen)
        assert fuente == "proxy_volumen"
        np.testing.assert_allclose(w, [0.25, 0.75])

        w, fuente = er.pesos_mercado(activos, fechas[-1])
        assert fuente == "equiponderado"
        np.testing.assert_allclose(w, [0.5, 0.5])

    def test_proveedor_incompleto_cae_al_proxy(self):
        fechas = pd.date_range("2019-01-31", periods=12, freq="ME")
        precios = pd.DataFrame(10.0, index=fechas, columns=["A", "B"])
        volumen = pd.DataFrame({"A": [100] * 12, "B": [100] * 12}, index=fechas)
        _, fuente = er.pesos_mercado(["A", "B"], fechas[-1], er.CapitalizacionEstatica({"A": 1.0}), precios, volumen)
        assert fuente == "proxy_volumen"


# ============================================================================
# Universo anidado
# ============================================================================


class TestAnidamiento:
    def test_anidamiento_exacto(self):
        tickers = [f"T{i}" for i in range(50)]
        u = er.generar_prioridad_universo(tickers, b=0)
        assert set(er.seleccionar_universo_anidado(tickers, u, 20)) <= set(er.seleccionar_universo_anidado(tickers, u, 40))

    def test_prioridad_depende_del_nombre_no_de_la_posicion(self):
        a = er.generar_prioridad_universo(["X", "Y", "Z"], b=3)
        b = er.generar_prioridad_universo(["Z", "W", "X"], b=3)
        assert a["X"] == b["X"] and a["Z"] == b["Z"]
        assert not er.generar_prioridad_universo(["X"], b=4).equals(er.generar_prioridad_universo(["X"], b=3))

    def test_universo_vacio_si_insuficientes_elegibles(self):
        tickers = [f"T{i}" for i in range(10)]
        assert er.seleccionar_universo_anidado(tickers[:5], er.generar_prioridad_universo(tickers, b=0), 20) == []

    def test_prioridad_invalida(self):
        with pytest.raises(ValueError):
            er.generar_prioridad_universo(["A"], b=0, metodo="otro")


# ============================================================================
# Elegibilidad point-in-time sin look-ahead
# ============================================================================


def _precios_mensuales(meses: int = 40, tickers=("A", "B", "C", "D")) -> pd.DataFrame:
    indice = pd.date_range("2015-01-01", periods=meses, freq="MS")
    rng = np.random.default_rng(0)
    return pd.DataFrame(100 * np.exp(rng.normal(0.005, 0.03, (meses, len(tickers))).cumsum(axis=0)),
                        index=indice, columns=list(tickers))


class TestElegibilidad:
    T = 12

    def _fecha(self, precios):
        return precios.index[24]  # t con historia suficiente

    def test_basico_cobertura_completa_y_membresia(self):
        precios = _precios_mensuales()
        t = self._fecha(precios)
        miembros = pd.DataFrame(True, index=precios.index, columns=precios.columns)
        assert list(er.elegibles_en_fecha(precios, miembros, t, self.T)) == ["A", "B", "C", "D"]

    def test_activo_que_deslista_despues_de_t_sigue_elegible(self):
        """Deslista en t+5 (precio NaN y salida del índice desde t+5): en t es elegible."""
        precios = _precios_mensuales()
        t = self._fecha(precios)
        precios.loc[precios.index[24 + 5]:, "B"] = np.nan
        miembros = pd.DataFrame(True, index=precios.index, columns=precios.columns)
        miembros.loc[precios.index[24 + 5]:, "B"] = False
        assert "B" in er.elegibles_en_fecha(precios, miembros, t, self.T)

    def test_activo_que_entra_al_indice_en_t_mas_1_no_es_elegible(self):
        precios = _precios_mensuales()
        t = self._fecha(precios)
        miembros = pd.DataFrame(True, index=precios.index, columns=precios.columns)
        miembros.loc[: precios.index[24], "C"] = False  # entra en t+1
        assert "C" not in er.elegibles_en_fecha(precios, miembros, t, self.T)
        assert "C" in er.elegibles_en_fecha(precios, miembros, precios.index[25], self.T)

    def test_no_exige_precio_en_t_ni_usa_datos_futuros(self):
        """Cambiar cualquier dato con fecha >= t no altera el resultado."""
        precios = _precios_mensuales()
        t = self._fecha(precios)
        miembros = pd.DataFrame(True, index=precios.index, columns=precios.columns)
        base = er.elegibles_en_fecha(precios, miembros, t, self.T)
        alterado = precios.copy()
        alterado.loc[t:, :] = np.nan
        miembros_alt = miembros.copy()
        miembros_alt.loc[t + pd.DateOffset(months=1):, :] = False
        assert list(er.elegibles_en_fecha(alterado, miembros_alt, t, self.T)) == list(base)

    def test_hueco_en_la_ventana_excluye(self):
        precios = _precios_mensuales()
        t = self._fecha(precios)
        precios.iloc[18, precios.columns.get_loc("D")] = np.nan  # un mes en t-T-1..t-1
        miembros = pd.DataFrame(True, index=precios.index, columns=precios.columns)
        assert "D" not in er.elegibles_en_fecha(precios, miembros, t, self.T)
        # con cobertura mínima laxa vuelve a entrar
        assert "D" in er.elegibles_en_fecha(precios, miembros, t, self.T, cobertura_min=0.8)

    def test_historia_insuficiente_excluye(self):
        precios = _precios_mensuales()
        precios.loc[: precios.index[14], "A"] = np.nan  # primera cotización: mes 15 → <T+1 meses hasta t
        miembros = pd.DataFrame(True, index=precios.index, columns=precios.columns)
        assert "A" not in er.elegibles_en_fecha(precios, miembros, self._fecha(precios), self.T)

    def test_membresia_como_mapping_y_mascara(self):
        precios = _precios_mensuales()
        t = self._fecha(precios)
        intervalos = {
            "A": [{"index": "SP500", "date_added": "2010-01-01", "date_removed": None}],
            "B": [{"index": "SP500", "date_added": "2010-01-01", "date_removed": "2016-01-01"}],
            "C": [{"index": "NASDAQ", "date_added": "2010-01-01", "date_removed": None}],
            "D": [{"index": "SP500", "date_added": str((t + pd.DateOffset(months=1)).date()), "date_removed": None}],
        }
        elegibles = er.elegibles_en_fecha(precios, intervalos, t, self.T)
        assert list(elegibles) == ["A"]
        mascara = er.construir_mascara_membresia(intervalos, precios.index, list(precios.columns))
        assert list(er.elegibles_en_fecha(precios, mascara, t, self.T)) == ["A"]

    def test_sin_membresias_no_filtra(self):
        precios = _precios_mensuales()
        assert len(er.elegibles_en_fecha(precios, None, self._fecha(precios), self.T)) == 4


# ============================================================================
# Turnover, coste y motor de rejilla
# ============================================================================


class TestTurnoverYCoste:
    def test_turnover_frente_a_pesos_a_la_deriva(self):
        previos = pd.Series({"A": 0.5, "B": 0.5})
        retornos = pd.Series({"A": 0.10, "B": -0.10})
        nuevos = pd.Series({"A": 0.5, "B": 0.5})
        turnover, deriva = er.calcular_turnover(nuevos, previos, retornos)
        # deriva: A=0.55, B=0.45 → turnover ½(0.05+0.05) = 0.05
        np.testing.assert_allclose(deriva.values, [0.55, 0.45])
        assert turnover == pytest.approx(0.05)

    def test_turnover_con_entradas_y_salidas(self):
        turnover, _ = er.calcular_turnover(pd.Series({"B": 1.0}), pd.Series({"A": 1.0}), pd.Series({"A": 0.0}))
        assert turnover == pytest.approx(1.0)

    def test_retorno_neto(self):
        tc, neto = er.calcular_retorno_neto(0.02, 0.5, coste_pb=10)
        assert tc == pytest.approx(0.0005)
        assert neto == pytest.approx(1.02 * (1 - 0.0005) - 1)
        assert all(np.isnan(v) for v in er.calcular_retorno_neto(np.nan, 0.5, 10))


def _datos_sinteticos(n_tickers: int = 30, meses: int = 80, seed: int = 0) -> er.DatosRejilla:
    rng = np.random.default_rng(seed)
    indice = pd.date_range("2014-01-01", periods=meses, freq="MS")
    tickers = [f"S{i:02d}" for i in range(n_tickers)]
    factor = rng.normal(0.006, 0.03, size=(meses, 1))
    ret = factor + rng.normal(0.0, 0.05, size=(meses, n_tickers))
    precios = pd.DataFrame(50 * np.cumprod(1 + ret, axis=0), index=indice, columns=tickers)
    volumen = pd.DataFrame(rng.uniform(1e5, 1e6, size=(meses, n_tickers)), index=indice, columns=tickers)
    rf = pd.Series(0.002, index=indice)
    return er.DatosRejilla(precios, volumen, rf, membresias=None)


def _config_pequena(**extra) -> er.ConfiguracionRejilla:
    base = {"valores_n": (5, 10), "valores_t": (12,), "n_replicas": 1, "replicas_michaud": 4,
            "fecha_oos_ini": "2018-01-01", "fecha_oos_fin": "2019-06-01", "max_replicas_con_pesos": 1}
    base.update(extra)
    return er.ConfiguracionRejilla(**base)


@pytest.fixture(scope="module")
def resultado():
    """Rejilla pequeña sobre datos sintéticos, calculada una sola vez para el módulo."""
    return er.ejecutar_rejilla(_datos_sinteticos(), _config_pequena())


class TestMotorRejilla:
    def test_esquema_de_columnas_y_numero_de_filas(self, resultado):
        columnas = {"b", "N", "T", "estrategia", "fecha", "r_bruto", "turnover", "tc", "r_neto", "n_activos_pos", "hhi",
                    "w_max", "dist_w_prev", "dist_w_mkt", "degenerate", "solver_status", "tiempo_ms", "conf_aislada",
                    "conf_conjunta", "dispersion_omega"}
        df = resultado.resultados_oos
        assert columnas <= set(df.columns)
        assert len(df) == 2 * 6 * 18  # N × estrategias × meses OOS
        assert set(df["estrategia"]) == set(er.ESTRATEGIAS_REJILLA)
        assert {"b", "N", "T", "estrategia", "fecha", "ticker", "w"} <= set(resultado.pesos_oos.columns)
        assert len(resultado.manifest_draws) == 30

    def test_pesos_suman_uno_y_primer_mes_sin_turnover(self, resultado):
        sumas = resultado.pesos_oos.groupby(["N", "estrategia", "fecha"])["w"].sum()
        np.testing.assert_allclose(sumas.to_numpy(), 1.0, atol=1e-5)
        primero = resultado.resultados_oos[resultado.resultados_oos["fecha"] == pd.Timestamp("2018-01-01")]
        assert (primero["turnover"] == 0.0).all()

    def test_coste_y_neto_consistentes(self, resultado):
        df = resultado.resultados_oos
        np.testing.assert_allclose(df["tc"], 10 * 1e-4 * df["turnover"])
        np.testing.assert_allclose(df["r_neto"], (1 + df["r_bruto"]) * (1 - df["tc"]) - 1)

    def test_igual_ponderacion_tiene_turnover_por_deriva(self, resultado):
        df = resultado.resultados_oos.query("estrategia == 'igual_ponderacion' and N == 5").sort_values("fecha")
        assert df["turnover"].iloc[1:].gt(0).all()  # 1/N rebalancea contra la deriva
        assert df["turnover"].iloc[1:].lt(0.5).all()

    def test_bl_prior_distancia_a_w_mkt_cero_y_contingencia_registrada(self, resultado):
        df = resultado.resultados_oos
        assert (df.loc[df["estrategia"] == "bl_prior", "dist_w_mkt"].abs() < 1e-12).all()
        assert df["degenerate"].dtype == bool
        assert df["solver_status"].notna().all()

    def test_reproducible_con_misma_semilla_y_distinto_con_otra(self):
        datos = _datos_sinteticos()
        a = er.ejecutar_rejilla(datos, _config_pequena(estrategias=("mpt_michaud",))).resultados_oos
        b = er.ejecutar_rejilla(datos, _config_pequena(estrategias=("mpt_michaud",))).resultados_oos
        c = er.ejecutar_rejilla(datos, _config_pequena(estrategias=("mpt_michaud",), semilla_base=7)).resultados_oos
        np.testing.assert_allclose(a["r_bruto"].to_numpy(), b["r_bruto"].to_numpy(), atol=1e-12)
        assert not np.allclose(a["r_bruto"].to_numpy(), c["r_bruto"].to_numpy())

    def test_tipo_retorno_log_difiere_de_simple_y_bootstrap_bloques_corre(self):
        datos = _datos_sinteticos()
        log = er.ejecutar_rejilla(datos, _config_pequena(estrategias=("mpt_muestral",))).resultados_oos
        simple = er.ejecutar_rejilla(
            datos, _config_pequena(estrategias=("mpt_muestral",), tipo_retorno_estimacion="simple")
        ).resultados_oos
        assert not np.allclose(log["r_bruto"].to_numpy(), simple["r_bruto"].to_numpy())
        bloques = er.ejecutar_rejilla(
            datos, _config_pequena(estrategias=("mpt_michaud",), metodo_bootstrap="bloques")
        ).resultados_oos
        assert bloques["solver_status"].eq("michaud_bootstrap").all()

    def test_cargador_inyectable_y_estrategia_desconocida(self):
        class Cargador:
            def cargar(self):
                return _datos_sinteticos(n_tickers=12)

        res = er.ejecutar_rejilla(Cargador(), _config_pequena(valores_n=(5,), estrategias=("igual_ponderacion",)))
        assert len(res.resultados_oos) == 18
        with pytest.raises(ValueError):
            er.ejecutar_rejilla(_datos_sinteticos(), _config_pequena(estrategias=("nada",)))

    def test_proveedor_de_capitalizacion_se_usa_en_bl_prior(self):
        datos = _datos_sinteticos(n_tickers=12)
        capitalizaciones = {t: float(i + 1) for i, t in enumerate(datos.precios_mensuales.columns)}
        datos.proveedor_capitalizacion = er.CapitalizacionEstatica(capitalizaciones)
        res = er.ejecutar_rejilla(datos, _config_pequena(valores_n=(5,), estrategias=("bl_prior",)))
        fila = res.pesos_oos.query("fecha == '2018-01-01'")
        caps = pd.Series(capitalizaciones)[fila["ticker"]]
        np.testing.assert_allclose(fila["w"].to_numpy(), (caps / caps.sum()).to_numpy(), atol=1e-6)


# ============================================================================
# Integración Strategy: modelos, crear_modelo_personalizado, presets
# ============================================================================


def _retornos_diarios(t: int = 300, n: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    indice = pd.bdate_range("2022-01-03", periods=t)
    datos = 0.0004 + rng.normal(0, 0.012, size=(t, 1)) + rng.normal(0, 0.008, size=(t, n))
    return pd.DataFrame(datos, index=indice, columns=[f"T{i}" for i in range(n)])


class TestIntegracionStrategy:
    def test_modelos_disponibles_solo_lista_lo_construible(self):
        disponibles = modelos_disponibles()
        for retorno in disponibles["returns_models"]:
            for riesgo in disponibles["risk_models"]:
                for optimizador in disponibles["optimizers"]:
                    crear_modelo_personalizado("m", retorno, riesgo, optimizador)
        for nuevo in ("bayes_stein", "bl_prior", "bl_momentum"):
            assert nuevo in disponibles["returns_models"]
        for nuevo in ("max_sharpe_qp", "michaud"):
            assert nuevo in disponibles["optimizers"]
        assert "fama_french_4" not in disponibles["returns_models"]  # no construible aquí
        with pytest.raises(ValueError):
            crear_modelo_personalizado("m", "fama_french_4", "ledoit_wolf", "max_sharpe")

    def test_crear_bayes_stein_max_sharpe_qp_y_optimizar(self):
        modelo = crear_modelo_personalizado(
            "bs", "bayes_stein", "ledoit_wolf", "max_sharpe_qp",
            returns_config={"frequency": 252}, optimizer_config={"risk_free_rate": 0.0, "basura": 1},
        )
        assert isinstance(modelo.returns_model, BayesSteinReturnsModel)
        assert isinstance(modelo.optimizer, MaxSharpeQPOptimizer)
        resultado = modelo.optimize(_retornos_diarios())
        assert sum(resultado["weights"].values()) == pytest.approx(1.0, abs=1e-6)
        assert resultado["optimizer_metadata"]["method"] in ("qp", "slsqp", "pypfopt", "gmv_contingencia")

    def test_crear_michaud_recibe_historico_y_config(self):
        modelo = crear_modelo_personalizado(
            "mi", "historical_mean", "ledoit_wolf", "michaud",
            optimizer_config={"n_replicas": 6, "metodo_bootstrap": "bloques", "longitud_media_bloque": 8, "seed": 1},
        )
        assert isinstance(modelo.optimizer, MichaudOptimizer)
        assert modelo.optimizer.metodo_bootstrap == "bloques" and modelo.optimizer.n_replicas == 6
        resultado = modelo.optimize(_retornos_diarios())
        assert sum(resultado["weights"].values()) == pytest.approx(1.0, abs=1e-6)
        assert resultado["optimizer_metadata"]["metodo_bootstrap"] == "bloques"

    def test_michaud_optimizer_sin_historico_lanza(self):
        mu = pd.Series([0.1, 0.2], index=["A", "B"])
        sigma = pd.DataFrame(np.eye(2) * 0.04, index=["A", "B"], columns=["A", "B"])
        with pytest.raises(ValueError, match="retornos_historicos"):
            MichaudOptimizer().optimize(mu, sigma)

    def test_bl_prior_con_market_caps_reproduce_pesos_de_mercado(self):
        retornos = _retornos_diarios()
        caps = {t: float(i + 1) * 1e9 for i, t in enumerate(retornos.columns)}
        modelo = crear_modelo_personalizado("bl", "bl_prior", "ledoit_wolf", "max_sharpe_qp",
                                            returns_config={"market_caps": caps})
        resultado = modelo.optimize(retornos)
        esperado = pd.Series(caps) / sum(caps.values())
        np.testing.assert_allclose([resultado["weights"][t] for t in retornos.columns], esperado.values, atol=1e-4)
        assert "avisos" not in resultado["optimizer_metadata"]

    def test_bl_prior_sin_market_caps_usa_proxy_y_avisa(self):
        retornos = _retornos_diarios()
        modelo = crear_modelo_personalizado("bl", "bl_prior", "ledoit_wolf", "max_sharpe_qp")
        resultado = modelo.optimize(retornos)
        avisos = resultado["optimizer_metadata"]["avisos"]
        assert len(avisos) == 1 and "market_caps" in avisos[0]
        assert modelo.returns_model.metadata["w_mkt_fuente"] == "equiponderado"
        # con precios y volumen usa el proxy por volumen
        precios = (1 + retornos).cumprod() * 100
        volumen = pd.DataFrame({c: [1e6 * (i + 1)] * len(retornos) for i, c in enumerate(retornos.columns)},
                               index=retornos.index)
        modelo.optimize(retornos, precios=precios, volumen=volumen)
        assert modelo.returns_model.metadata["w_mkt_fuente"] == "proxy_volumen"

    def test_bl_momentum_modelo_y_covarianza_predictiva(self):
        retornos = _retornos_diarios(t=40, n=4)
        modelo = BLMomentumReturnsModel(frequency=252, ventana_senal=30, omitir_ultimas=5)
        mu = modelo.calculate(retornos)
        assert isinstance(mu, pd.Series) and list(mu.index) == list(retornos.columns)
        sigma = LedoitWolfShrinkageModel(252).calculate(retornos)
        total = modelo.covarianza_predictiva(sigma)
        assert (np.diag(total.values) >= np.diag(sigma.values)).all()
        with pytest.raises(RuntimeError):
            BLMomentumReturnsModel().covarianza_predictiva(sigma)
        crear_modelo_personalizado("m", "bl_momentum", "ledoit_wolf", "max_sharpe_qp").optimize(retornos)

    def test_bl_prior_model_calcula_pi(self):
        retornos = _retornos_diarios()
        caps = {t: 1.0 for t in retornos.columns}
        modelo = BLPriorReturnsModel(risk_aversion=2.5, frequency=252, market_caps=caps)
        sigma = LedoitWolfShrinkageModel(252).calculate(retornos)
        pi = modelo.calculate(retornos, cov_matrix=sigma)
        np.testing.assert_allclose(pi.values, 2.5 * sigma.values @ np.full(5, 0.2))

    @pytest.mark.parametrize("nombre", ["tfm_bayes_stein", "tfm_michaud", "tfm_bl_prior"])
    def test_presets_tfm_registrados_y_optimizan(self, nombre):
        assert nombre in listar_presets()
        resultado = obtener_preset(nombre).optimize(_retornos_diarios())
        assert sum(resultado["weights"].values()) == pytest.approx(1.0, abs=1e-6)
        assert all(w >= -1e-9 for w in resultado["weights"].values())

    def test_preset_tfm_bl_prior_avisa_sin_market_caps(self):
        resultado = obtener_preset("tfm_bl_prior").optimize(_retornos_diarios())
        assert resultado["optimizer_metadata"]["avisos"]

    def test_preset_pmpt_cvar_usa_optimizador_cvar(self):
        modelo = obtener_preset("pmpt_cvar")
        assert isinstance(modelo.optimizer, PMPTMinCVaROptimizer)
        resultado = modelo.optimize(_retornos_diarios())
        assert "cvar" in resultado
        assert sum(resultado["weights"].values()) == pytest.approx(1.0, abs=1e-3)
        # distinto de la semivarianza + min_volatility que usaba antes
        assert resultado["optimizer_metadata"]["method"] in ("pypfopt_ecvar", "slsqp")


# ============================================================================
# Volatilidad objetivo como restricción (2026-11-19)
# ============================================================================


def _mu_sigma_tres_activos():
    """Tres activos: volatilidades 10 %, 20 % y 30 %, rentabilidades 4 %, 8 % y 12 %, ρ = 0,2."""
    vol = np.array([0.10, 0.20, 0.30])
    corr = np.full((3, 3), 0.2) + 0.8 * np.eye(3)
    return np.array([0.04, 0.08, 0.12]), np.outer(vol, vol) * corr


class TestVolatilidadObjetivo:
    def test_restriccion_activa_alcanza_la_volatilidad(self):
        mu, sigma = _mu_sigma_tres_activos()
        pesos, estado = er.optimizar_volatilidad_objetivo(mu, sigma, 0.15)
        assert estado == "volatilidad_objetivo"
        assert pesos.sum() == pytest.approx(1.0, abs=1e-8) and np.all(pesos >= -1e-9)
        assert np.sqrt(pesos @ sigma @ pesos) == pytest.approx(0.15, abs=1e-4)

    def test_mas_volatilidad_mas_rentabilidad(self):
        mu, sigma = _mu_sigma_tres_activos()
        rentabilidades = [mu @ er.optimizar_volatilidad_objetivo(mu, sigma, v)[0] for v in (0.11, 0.15, 0.2, 0.25)]
        assert rentabilidades == sorted(rentabilidades)

    def test_objetivo_por_debajo_de_la_minima_varianza_devuelve_gmv(self):
        mu, sigma = _mu_sigma_tres_activos()
        pesos, estado = er.optimizar_volatilidad_objetivo(mu, sigma, 0.02)
        assert estado == "gmv_volatilidad_inalcanzable"
        np.testing.assert_allclose(pesos, er.optimizar_min_varianza(sigma), atol=1e-4)

    def test_objetivo_holgado_y_tope_de_peso(self):
        mu, sigma = _mu_sigma_tres_activos()
        pesos, estado = er.optimizar_volatilidad_objetivo(mu, sigma, 0.9)
        assert estado == "volatilidad_objetivo_holgada"
        np.testing.assert_allclose(pesos, [0, 0, 1], atol=1e-4)
        pesos, _ = er.optimizar_volatilidad_objetivo(mu, sigma, 0.9, peso_maximo=0.5)
        np.testing.assert_allclose(pesos, [0, 0.5, 0.5], atol=1e-4)

    def test_max_sharpe_qp_con_volatilidad_objetivo(self):
        modelo = crear_modelo_personalizado(
            "bs", "bayes_stein", "ledoit_wolf", "max_sharpe_qp",
            returns_config={"frequency": 252}, optimizer_config={"volatilidad_objetivo": 0.12},
        )
        resultado = modelo.optimize(_retornos_diarios())
        assert resultado["optimizer_metadata"]["volatilidad_objetivo"] == 0.12
        assert resultado["optimizer_metadata"]["method"] in ("volatilidad_objetivo", "volatilidad_objetivo_holgada",
                                                             "gmv_volatilidad_inalcanzable")
        if resultado["optimizer_metadata"]["method"] == "volatilidad_objetivo":
            assert resultado["volatility"] == pytest.approx(0.12, abs=1e-3)

    def test_black_litterman_con_efficient_risk(self):
        # BL (prior de mercado) + restricción de volatilidad: la «versión BL» de fijar la volatilidad
        retornos = _retornos_diarios()
        caps = {t: float(i + 1) * 1e9 for i, t in enumerate(retornos.columns)}
        modelo = crear_modelo_personalizado(
            "bl", "bl_prior", "ledoit_wolf", "efficient_risk",
            returns_config={"market_caps": caps}, optimizer_config={"target_volatility": 0.16},
        )
        resultado = modelo.optimize(retornos)
        assert sum(resultado["weights"].values()) == pytest.approx(1.0, abs=1e-6)
        assert resultado["optimizer_metadata"]["volatilidad_objetivo"] == 0.16
        assert resultado["volatility"] <= 0.16 + 1e-3 or resultado["optimizer_metadata"]["method"] == "gmv_volatilidad_inalcanzable"

    def test_efficient_risk_sin_objetivo_lanza(self):
        modelo = crear_modelo_personalizado("m", "historical_mean", "ledoit_wolf", "efficient_risk")
        with pytest.raises(ValueError, match="target_volatility"):
            modelo.optimize(_retornos_diarios())

    def test_michaud_con_volatilidad_objetivo(self):
        modelo = crear_modelo_personalizado(
            "mi", "historical_mean", "ledoit_wolf", "michaud",
            optimizer_config={"n_replicas": 6, "seed": 1, "volatilidad_objetivo": 0.12},
        )
        resultado = modelo.optimize(_retornos_diarios())
        assert sum(resultado["weights"].values()) == pytest.approx(1.0, abs=1e-6)
        assert resultado["optimizer_metadata"]["volatilidad_objetivo"] == 0.12
