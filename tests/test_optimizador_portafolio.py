"""
test_optimizador_portafolio.py — Tests de ypyquant.optimization.optimizador_portafolio.

Responsabilidad:
    - Propiedades de los optimizadores (sobre retornos sintéticos con varias semillas):
      pesos que suman 1, sin pesos negativos (no hay cortos), límite max_weight respetado,
      la mínima varianza no es más volátil que pesos iguales, frontera con un único óptimo.
    - Objetivos financieros: la tasa encontrada reproduce el valor futuro pedido; validación
      de retornos irreales; Capital Market Line.
    - PMPT: semivarianza, CVaR ≤ VaR, PMPT con objetivos financieros.
    - Servicio PortfolioOptimizationService con cargador inyectado (API de GrowSave sin BD),
      create_custom_model con 'historical_mean' (fallaba en GrowSave por el enum) y presets.
Datos sintéticos, sin BD ni red.

Se relaciona con: src/ypyquant/optimization/optimizador_portafolio.py, portfolio_models.py.
"""

import numpy as np
import pandas as pd
import pytest

from ypyquant.optimization import optimizador_portafolio as opt
from ypyquant.optimization.optimizador_portafolio import PortfolioOptimizationService

TICKERS = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']


def _retornos(semilla: int, n: int = 500) -> pd.DataFrame:
    """Retornos diarios con correlación y medias distintas por activo."""
    rng = np.random.default_rng(semilla)
    factor = rng.normal(0, 0.01, n)
    datos = {
        t: 0.0002 * (i + 1) + 0.5 * factor + rng.normal(0, 0.008 + 0.003 * i, n)
        for i, t in enumerate(TICKERS)
    }
    return pd.DataFrame(datos, index=pd.bdate_range('2022-01-03', periods=n))


def _comprobar_pesos(pesos: dict, max_peso: float = 1.0):
    valores = np.array(list(pesos.values()))
    assert valores.sum() == pytest.approx(1.0, abs=1e-4)
    assert (valores >= -1e-8).all()
    assert (valores <= max_peso + 1e-6).all()


@pytest.mark.parametrize('semilla', [1, 2, 3])
@pytest.mark.parametrize('objetivo', ['max_sharpe', 'min_vol', 'equal_weight', 'risk_parity'])
def test_pesos_validos_por_objetivo(semilla, objetivo):
    res = opt.optimizar_portafolio(_retornos(semilla), objetivo)
    _comprobar_pesos(res['weights'])
    assert set(res['weights']) == set(TICKERS)


@pytest.mark.parametrize('semilla', [4, 5])
@pytest.mark.parametrize('objetivo', ['max_sharpe', 'min_vol', 'risk_parity'])
def test_respeta_max_weight(semilla, objetivo):
    res = opt.optimizar_portafolio(_retornos(semilla), objetivo, {'max_weight': 0.3})
    _comprobar_pesos(res['weights'], max_peso=0.3)


@pytest.mark.parametrize('semilla', [6, 7])
def test_minima_varianza_no_supera_pesos_iguales(semilla):
    r = _retornos(semilla)
    assert opt.optimizar_portafolio(r, 'min_vol')['volatility'] <= opt.optimizar_portafolio(r, 'equal_weight')['volatility'] + 1e-9


def test_risk_parity_iguala_contribuciones():
    r = _retornos(8)
    cov = r.cov() * 252
    w = np.array(list(opt.optimizar_portafolio(r, 'risk_parity')['weights'].values()))
    contrib = w * (cov.values @ w) / np.sqrt(w @ cov.values @ w)
    assert contrib.std() / contrib.mean() < 0.05


def test_objetivo_desconocido():
    with pytest.raises(ValueError):
        opt.optimizar_portafolio(_retornos(1), 'kelly')


def test_frontera_eficiente():
    puntos = opt.calcular_frontera_eficiente(_retornos(9), num_points=12)
    assert len(puntos) == 12
    assert sum(p['is_optimal'] for p in puntos) == 1
    for p in puntos:
        _comprobar_pesos(p['weights'])
    optimo = next(p for p in puntos if p['is_optimal'])
    assert optimo['sharpe_ratio'] == max(p['sharpe_ratio'] for p in puntos)


def test_volatilidad_objetivo():
    r = _retornos(10)
    res = opt.optimizar_volatilidad_objetivo(r, 0.15)
    _comprobar_pesos(res['weights'])
    assert res['volatility'] <= 0.15 + 1e-4


def test_cml_mezcla_con_liquidez():
    res = opt.ajustar_portafolio_a_volatilidad({'A': 0.4, 'B': 0.6}, 0.15, 0.10, 0.12, 0.02)
    assert res['cash_weight'] == pytest.approx(1 / 3)
    assert sum(res['weights'].values()) + res['cash_weight'] == pytest.approx(1.0)
    assert res['expected_return'] == pytest.approx(2 / 3 * 0.12 + 1 / 3 * 0.02)
    with pytest.raises(ValueError):
        opt.ajustar_portafolio_a_volatilidad({'A': 1.0}, 0.10, 0.20, 0.08, 0.02)


@pytest.mark.parametrize('capital,aportacion,meses,objetivo', [
    (10_000, 500, 36, 30_000), (0, 200, 120, 40_000), (50_000, 0, 60, 70_000),
])
def test_tasa_objetivo_reproduce_el_valor_futuro(capital, aportacion, meses, objetivo):
    res = opt.calcular_retorno_objetivo_desde_metas(capital, aportacion, meses, objetivo)
    r = res['target_return_monthly']
    vf = capital * (1 + r) ** meses + aportacion * (((1 + r) ** meses - 1) / r)
    assert vf == pytest.approx(objetivo, rel=1e-6)
    assert res['target_return_annual'] == pytest.approx((1 + r) ** 12 - 1)


def test_metas_sin_ganancia_y_errores():
    assert opt.calcular_retorno_objetivo_desde_metas(10_000, 500, 12, 12_000)['target_return_annual'] == 0.0
    assert opt.calcular_retorno_objetivo_desde_metas(-1, 0, 0, 0)['validation_errors']


def test_validar_retorno_objetivo():
    r = _retornos(11)
    assert not opt.validar_retorno_objetivo(0.35, r)['is_valid']
    assert opt.validar_retorno_objetivo(0.05, r)['is_valid']


def test_semivarianza_y_cvar():
    r = _retornos(12)
    semi = opt.calcular_semivarianza(r)
    assert np.all(np.diag(semi) <= np.diag(r.cov() * 252) + 1e-12)
    w = np.full(len(TICKERS), 1 / len(TICKERS))
    cvar = opt.calcular_cvar(r, w, 0.95, time_horizon_days=252)
    assert cvar['cvar'] <= cvar['var']
    assert cvar['cvar_horizon'] == pytest.approx(cvar['cvar'])  # √252 = horizonte de un año


@pytest.mark.parametrize('objetivo', ['min_semivariance', 'max_sortino', 'min_cvar'])
def test_pmpt_pesos_validos(objetivo):
    res = opt.optimizar_pmpt(_retornos(13), objetivo, target_return=0.04)
    _comprobar_pesos(res['weights'])
    assert res['optimization_method'] == objetivo


def test_pmpt_desde_objetivos_financieros():
    res = opt.optimizar_pmpt(_retornos(14), capital_inicial=10_000, aportacion_mensual=300, meses=36,
                             objetivo_final=22_000, validate_target=False)
    assert res['target_return_info']['target_return_annual'] == pytest.approx(res['target_return'])
    with pytest.raises(ValueError):
        opt.optimizar_pmpt(_retornos(14), 'otro')


# ---------------------------------------------------------------------------
# Servicio con cargador inyectado
# ---------------------------------------------------------------------------

def _cargador(semilla: int = 20):
    r = _retornos(semilla)
    precios = 100 * np.exp(r.cumsum())

    def cargar(ticker, inicio, fin):
        if ticker not in precios:
            return pd.DataFrame()
        serie = precios[ticker].loc[inicio:fin]
        return pd.DataFrame({'close': serie, 'adjusted_close': serie})
    return cargar


def test_servicio_con_cargador():
    s = PortfolioOptimizationService(cargador_precios=_cargador())
    res = s.optimize_portfolio(TICKERS + ['NOEXISTE'], '2022-01-01', '2023-12-31', 'max_sharpe')
    _comprobar_pesos(res['weights'])
    assert 'NOEXISTE' not in res['weights']
    assert len(s.calculate_efficient_frontier(TICKERS, '2022-01-01', '2023-12-31', 5)) == 5
    with pytest.raises(ValueError):
        s.optimize_portfolio(['NOEXISTE'], '2022-01-01', '2023-12-31')
    with pytest.raises(RuntimeError):
        PortfolioOptimizationService().optimize_portfolio(TICKERS, '2022-01-01', '2023-12-31')


def test_metricas_de_pesos_dados():
    s = PortfolioOptimizationService(cargador_precios=_cargador())
    w = np.full(len(TICKERS), 0.2)
    res = s.calcular_metricas_cartera(TICKERS, w, '2022-01-01', '2023-12-31')
    esperado = s.optimize_portfolio(TICKERS, '2022-01-01', '2023-12-31', 'equal_weight')
    assert res['expected_return'] == pytest.approx(esperado['expected_return'])
    assert res['volatility'] == pytest.approx(esperado['volatility'])


@pytest.mark.parametrize('tipo_retornos', ['historical_mean', 'historical_median'])
def test_modelo_personalizado_historico(tipo_retornos):
    """En GrowSave create_custom_model fallaba siempre: ReturnModelType no tenía HISTORICAL_MEAN."""
    s = PortfolioOptimizationService(cargador_precios=_cargador())
    modelo = s.create_custom_model('m', tipo_retornos, 'ledoit_wolf', 'max_sharpe')
    res = s.optimize_advanced(TICKERS, '2022-01-01', '2023-12-31', modelo)
    assert sum(res['weights'].values()) == pytest.approx(1.0, abs=1e-3)
    assert res['date_range']['days'] > 400
    with pytest.raises(ValueError):
        s.create_custom_model('m', 'nada', 'ledoit_wolf', 'max_sharpe')


def test_gancho_de_datos_externos_y_comparacion():
    vistos = []

    class ServicioApp(PortfolioOptimizationService):
        def _obtener_datos_modelo(self, model, start_date, end_date):
            vistos.append(model.name)
            return {}

    s = ServicioApp(cargador_precios=_cargador())
    modelos = [s.create_custom_model('a', 'historical_mean', 'sample_covariance', 'min_volatility'),
               s.create_custom_model('b', 'historical_mean', 'ledoit_wolf', 'risk_parity')]
    res = s.compare_models(TICKERS, '2022-01-01', '2023-12-31', modelos)
    assert vistos == ['a', 'b']
    assert sorted(r['rank'] for r in res.values()) == [1, 2]


def test_presets_y_modelos_disponibles():
    s = PortfolioOptimizationService()
    assert 'markowitz' in s.list_available_presets()
    assert 'historical_mean' in s.get_available_models()['returns_models']
    assert s.get_preset_details('markowitz')['tipo'] == 'tradicional'
