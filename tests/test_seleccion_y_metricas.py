"""
test_seleccion_y_metricas.py — Tests de AssetSelectionService y de PerformanceAnalyzer.compare_to_benchmark.

Responsabilidad:
    - AssetSelectionService con cargador inyectado (API de GrowSave sin BD): screening por
      Sharpe/momentum/volatilidad, filtros técnicos y fundamentales, propiedad de la
      diversificación greedy (ningún par supera la correlación máxima), reproducibilidad de
      diversify_distance con la semilla de producción, universo histórico inyectado y
      screen_custom con columnas en minúscula (en GrowSave fallaba siempre por 'Close').
    - compare_to_benchmark: beta = cov/var, alpha de Jensen CAPM (la que usa GrowSave) y
      alpha de regresión (la que devolvía ypyquant 0.2.0).
Datos sintéticos, sin BD ni red.

Se relaciona con: src/ypyquant/portfolio/asset_selection.py, performance_metrics.py.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ypyquant.portfolio.asset_selection import AssetSelectionService, diversify_distance
from ypyquant.portfolio.performance_metrics import PerformanceAnalyzer

TICKERS = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF']
FECHAS = pd.bdate_range('2021-01-01', '2023-12-29')


def _precios(semilla: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    comun = rng.normal(0, 0.01, len(FECHAS))
    return pd.DataFrame({
        t: 100 * np.exp(np.cumsum(0.0003 * i + (0.9 if i < 3 else 0.1) * comun + rng.normal(0, 0.01, len(FECHAS))))
        for i, t in enumerate(TICKERS)
    }, index=FECHAS)


PRECIOS = _precios()


def cargador(ticker, inicio, fin):
    """Imita get_ticker_data de GrowSave (columnas en minúscula)."""
    serie = PRECIOS[ticker].loc[inicio:fin]
    return pd.DataFrame({'close': serie, 'adjusted_close': serie})


@pytest.fixture
def servicio():
    return AssetSelectionService(
        cargador_precios=cargador,
        proveedor_universo=lambda indice, ini, fin, solo_supervivientes: TICKERS[:4],
        proveedor_fundamentales=lambda t: {'market_cap': 1e9 * (TICKERS.index(t) + 1), 'pe_ratio': 15},
    )


def test_screening_basico_ordenado(servicio):
    sharpe = servicio.screen_by_sharpe(TICKERS, '2021-01-01', '2023-12-29', top_n=3)
    assert len(sharpe) == 3
    assert [r['sharpe_ratio'] for r in sharpe] == sorted((r['sharpe_ratio'] for r in sharpe), reverse=True)
    vol = servicio.screen_by_volatility(TICKERS, '2021-01-01', '2023-12-29', top_n=6)
    assert [r['annualized_volatility'] for r in vol] == sorted(r['annualized_volatility'] for r in vol)
    mom = servicio.screen_by_momentum(TICKERS, '2021-01-01', '2023-12-29', top_n=6)
    assert set(mom[0]) == {'ticker', 'total_return', 'annualized_volatility', 'sharpe_ratio', 'momentum_score'}


def test_universo_historico_inyectado(servicio):
    res = servicio.screen_by_sharpe(None, '2021-01-01', '2023-12-29', index_name='SP500')
    assert {r['ticker'] for r in res} == set(TICKERS[:4])
    with pytest.raises(ValueError):
        servicio.screen_by_sharpe(None, None, None, index_name='SP500')
    with pytest.raises(RuntimeError):
        AssetSelectionService(cargador).get_historical_universe('SP500', '2021-01-01', '2022-01-01')


@pytest.mark.parametrize('max_corr', [0.3, 0.5, 0.8])
def test_greedy_respeta_correlacion_maxima(servicio, max_corr):
    elegidos = servicio.diversify_greedy(TICKERS, '2021-01-01', '2023-12-29', max_corr)
    corr = servicio.calculate_correlation_matrix(elegidos, '2021-01-01', '2023-12-29')
    fuera_diagonal = corr.abs().values[~np.eye(len(elegidos), dtype=bool)]
    assert (fuera_diagonal < max_corr).all()
    assert elegidos[0] == TICKERS[0]


def test_distance_reproduce_la_semilla_de_produccion():
    """GrowSave usaba np.random.seed(42) + np.random.choice; RandomState(42) da la misma secuencia."""
    retornos = PRECIOS.pct_change().dropna()
    np.random.seed(42)
    esperado = None
    mejor = -1.0
    dist = 1 - retornos.corr().abs()
    for _ in range(100):
        cand = np.random.choice(retornos.columns, size=3, replace=False).tolist()
        d = np.mean([dist.loc[a, b] for i, a in enumerate(cand) for b in cand[i + 1:]])
        if d > mejor:
            mejor, esperado = d, cand
    assert diversify_distance(retornos, n_activos=3) == esperado


def test_clustering_y_mixed(servicio):
    reps = servicio.diversify_clustering(TICKERS, '2021-01-01', '2023-12-29', n_clusters=2)
    assert len(reps) == 2
    scores = {t: -i for i, t in enumerate(TICKERS)}
    mixto = servicio.diversify_mixed(TICKERS, scores, '2021-01-01', '2023-12-29', 0.5, top_n=2)
    assert mixto[0] == 'AAA' and len(mixto) <= 2


def test_screen_custom_con_columnas_minuscula(servicio):
    """En GrowSave ohlcv.get('adjusted_close', ..., ohlcv['Close']) lanzaba KeyError: 0 resultados siempre."""
    res = servicio.screen_custom(TICKERS, '2021-01-01', '2023-12-29', {'min_sharpe': -10})
    assert len(res) == len(TICKERS)


def test_filtros_tecnicos_y_fundamentales(servicio):
    res = servicio.screen_by_filters(TICKERS, '2021-01-01', '2023-12-29',
                                     {'market_cap_min': 3.5e9, 'max_drawdown_max': -1.0})
    assert {r['ticker'] for r in res} == {'DDD', 'EEE', 'FFF'}
    assert all(r['market_cap'] >= 3.5e9 for r in res)
    todos = servicio.screen_by_filters(TICKERS, '2021-01-01', '2023-12-29', {}, top_n=4)
    assert len(todos) == 4
    assert set(servicio.get_screening_presets()) == {'momentum_growth', 'low_volatility', 'high_sharpe', 'balanced'}


def test_matriz_retornos_vacia_si_no_hay_datos():
    s = AssetSelectionService(cargador_precios=lambda t, a, b: pd.DataFrame())
    assert s.diversify_greedy(['X'], '2021-01-01', '2022-01-01') == []
    assert s.calculate_correlation_matrix(['X'], '2021-01-01', '2022-01-01').empty


# ---------------------------------------------------------------------------
# compare_to_benchmark
# ---------------------------------------------------------------------------

def test_beta_y_alphas():
    rng = np.random.default_rng(0)
    mercado = pd.Series(rng.normal(0.0004, 0.01, 750), index=pd.bdate_range('2021-01-01', periods=750))
    cartera = 0.0002 + 1.3 * mercado + pd.Series(rng.normal(0, 0.004, 750), index=mercado.index)
    rf = 0.03
    res = PerformanceAnalyzer.compare_to_benchmark(cartera, mercado, risk_free_rate=rf)

    beta_cov = cartera.cov(mercado) / mercado.var()
    assert res['beta'] == pytest.approx(beta_cov, rel=1e-10)
    alpha_jensen = res['portfolio'].annualized_return - (rf + beta_cov * (res['benchmark'].annualized_return - rf))
    assert res['alpha'] == pytest.approx(alpha_jensen, rel=1e-10)
    intercepto = cartera.mean() - beta_cov * mercado.mean()
    assert res['alpha_regresion'] == pytest.approx(intercepto * 252, rel=1e-8)
    assert 0 <= res['beta_pvalue'] < 1e-6 and res['beta_stderr'] > 0
    assert PerformanceAnalyzer.compare_to_benchmark(cartera, cartera.shift(5000))['error']


def test_registro_fechas_universo():
    vistos = {}

    def universo(indice, ini, fin, solo):
        vistos.update(indice=indice, ini=ini, solo=solo)
        return ['AAA']

    AssetSelectionService(cargador, proveedor_universo=universo).get_historical_universe(
        'NASDAQ100', '2021-01-01', '2022-01-01', avoid_survivorship_bias=True
    )
    assert vistos == {'indice': 'NASDAQ100', 'ini': datetime(2021, 1, 1), 'solo': False}
