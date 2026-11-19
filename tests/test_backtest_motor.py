"""
test_backtest_motor.py — Tests del motor de backtest unificado (ypyquant.portfolio.backtest_engine).

Responsabilidad:
    - Port de GrowSave tests/test_backtest_motor.py (regresiones del 2026-11-15/17):
        · get_ticker_data devuelve columnas en minúscula ('close', 'adjusted_close') y el motor
          buscaba 'Close': ningún día tenía precio (0 rebalanceos, curva vacía);
        · la última compra de cada rebalanceo se rechazaba por «cash insuficiente» y la
          cartera se quedaba casi toda en liquidez;
        · la primera optimización solo veía datos desde la fecha de inicio: ahora se cargan
          VENTANA_ESTIMACION_DIAS de historia previa;
        · benchmark en la divisa del usuario con los mismos tipos de cambio que la cartera.
      En GrowSave se parcheaba `get_ticker_data` del módulo; aquí se inyecta un cargador.
    - Propiedades del motor: pesos iguales con rebalanceo diario y sin costes = media de los
      rendimientos; efectivo nunca negativo; comisiones = suma de las de cada operación;
      reutilización del motor; alias en español de la API 0.2.0 (scripts/run_fase2.py).
Datos sintéticos, sin BD ni red.

Se relaciona con: src/ypyquant/portfolio/backtest_engine.py, src/ypyquant/data/matriz_precios.py.
"""

import numpy as np
import pandas as pd
import pytest

from ypyquant.portfolio import backtest_engine as motor
from ypyquant.portfolio.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    PortfolioBacktestService,
    format_backtest_result,
)

# Dos activos con crecimiento diario constante: A dobla y B sube un 50 % en el periodo
FECHAS = pd.bdate_range('2020-01-01', '2026-01-01')
CRECIMIENTO = {'AAA': 2.0, 'BBB': 1.5}
INICIO, FIN = '2023-01-02', '2025-12-31'


class CargadorSimulado:
    """Imita get_ticker_data de GrowSave: columnas en minúscula e índice de fechas. Registra las cargas."""

    def __init__(self):
        self.cargas: list[str] = []

    def __call__(self, ticker, inicio, fin):
        self.cargas.append(inicio)
        dentro = (FECHAS >= pd.Timestamp(INICIO)) & (FECHAS <= pd.Timestamp(FIN))
        n = dentro.sum()
        tasa = CRECIMIENTO[ticker] ** (1 / (n - 1))
        # Precio plano antes del periodo; crecimiento constante dentro
        exponente = np.clip(np.cumsum(dentro) - 1, 0, None)
        precio = 100 * tasa ** exponente
        df = pd.DataFrame({'close': precio, 'adjusted_close': precio}, index=FECHAS)
        return df[(df.index >= pd.Timestamp(inicio)) & (df.index <= pd.Timestamp(fin))]


@pytest.fixture
def cargador():
    return CargadorSimulado()


@pytest.fixture
def ejecutar(cargador):
    def _ejecutar(pesos):
        cfg = BacktestConfig(start_date=INICIO, end_date=FIN, initial_capital=10000,
                             rebalance_frequency='quarterly', commission_rate=0.001,
                             slippage_pct=0.001, user_currency='USD')
        e = BacktestEngine(cfg, cargador_precios=cargador)
        return e, e.run_backtest(lambda fecha, datos: pesos, list(CRECIMIENTO))
    return _ejecutar


# ---------------------------------------------------------------------------
# Port de GrowSave
# ---------------------------------------------------------------------------

def test_un_activo_replica_su_crecimiento(ejecutar):
    # 100 % en A, que dobla: ≈ +100 % menos costes (comisión + slippage de entrada)
    e, r = ejecutar({'AAA': 1.0, 'BBB': 0.0})
    assert r.total_return_pct == pytest.approx(1.0, abs=0.02)
    assert e.cash == pytest.approx(0.0, abs=20)  # totalmente invertida


def test_mitad_y_mitad_invierte_todo(ejecutar):
    # 50/50 con rebalanceo: entre el +50 % de B y el +100 % de A, sin liquidez ociosa
    e, r = ejecutar({'AAA': 0.5, 'BBB': 0.5})
    assert 0.6 < r.total_return_pct < 0.8
    assert e.cash < 100


def test_carga_historia_previa_para_estimar(ejecutar, cargador):
    ejecutar({'AAA': 1.0, 'BBB': 0.0})
    esperado = (pd.Timestamp(INICIO) - pd.Timedelta(days=motor.VENTANA_ESTIMACION_DIAS)).strftime('%Y-%m-%d')
    assert esperado in cargador.cargas


class _CambioSimulado:
    """Tipos de cambio falsos: USD→EUR = 0,80 + 0,0001 por día desde INICIO (EUR→USD es la inversa)."""

    def tasa_usd_eur(self, fecha):
        dias = (pd.Timestamp(fecha) - pd.Timestamp(INICIO)).days
        return 0.80 + 0.0001 * dias

    def convert_amount(self, importe, origen, destino, fecha=None):
        if origen == destino:
            return importe
        tasa = self.tasa_usd_eur(fecha)
        return importe * tasa if (origen, destino) == ('USD', 'EUR') else importe / tasa


def _benchmark(aportacion, divisa):
    servicio = PortfolioBacktestService(cargador_precios=CargadorSimulado())
    return servicio._backtest_benchmark(
        'AAA', INICIO, FIN, 10000, aportacion, user_currency=divisa, servicio_cambio=_CambioSimulado()
    )


def test_benchmark_en_eur_es_usd_por_tipo_de_cambio():
    """Sin aportaciones: benchmark EUR = benchmark USD (capital en USD equivalente) × tipo de cambio de cada fecha."""
    cambio = _CambioSimulado()
    en_eur = _benchmark(0.0, 'EUR')['equity_curve_df']['equity']
    capital_usd = cambio.convert_amount(10000, 'EUR', 'USD', pd.Timestamp(INICIO).date())
    precios = CargadorSimulado()('AAA', INICIO, FIN)['close']
    en_usd = capital_usd / precios.iloc[0] * precios
    esperado = pd.Series(
        [cambio.convert_amount(v, 'USD', 'EUR', i.date()) for i, v in en_usd.items()], index=en_usd.index
    )
    pd.testing.assert_series_equal(en_eur, esperado, check_names=False, check_freq=False)
    assert not np.allclose(_benchmark(0.0, 'USD')['equity_curve_df']['equity'].values, en_eur.values)


def test_benchmark_con_aportaciones_en_eur_invierte_en_usd_al_cambio_del_dia():
    """Con DCA, el total aportado se mide en EUR (capital + aportaciones) y la curva está en EUR."""
    r = _benchmark(500, 'EUR')
    meses = len(pd.date_range(INICIO, FIN, freq='MS'))
    assert r['total_invested'] == pytest.approx(10000 + 500 * meses, rel=0.05)
    assert r['initial_capital'] == 10000
    assert 10000 + 500 * meses < r['final_capital'] < 3 * (10000 + 500 * meses) / 0.8


# ---------------------------------------------------------------------------
# Propiedades del motor
# ---------------------------------------------------------------------------

def _historico_aleatorio(semilla: int, n_activos: int = 4) -> dict[str, pd.DataFrame]:
    """Precios log-normales reproducibles con columna 'Close'."""
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range('2021-01-01', '2022-12-30')
    return {
        f'T{i}': pd.DataFrame(
            {'Close': 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(fechas))))}, index=fechas
        )
        for i in range(n_activos)
    }


@pytest.mark.parametrize('semilla', [1, 2, 3])
def test_pesos_iguales_diario_sin_costes_es_media_de_rendimientos(semilla):
    """Con rebalanceo diario y sin costes, el rendimiento diario ≈ media de los rendimientos simples."""
    datos = _historico_aleatorio(semilla)
    n = len(datos)
    cfg = BacktestConfig(start_date='2021-01-04', end_date='2022-12-30', initial_capital=1e9,
                         rebalance_frequency='daily', commission_rate=0.0, slippage_pct=0.0)
    r = BacktestEngine(cfg).run_backtest(lambda f, d: {t: 1 / n for t in d}, datos)
    rend_cartera = r.equity_curve['equity'].pct_change().dropna()
    rend_medio = pd.DataFrame({t: df['Close'] for t, df in datos.items()}).pct_change().mean(axis=1)
    rend_medio = rend_medio.reindex(rend_cartera.index)
    # El umbral de 10 unidades por operación deja una deriva mínima frente a 1e9 de capital
    assert np.max(np.abs(rend_cartera.values - rend_medio.values)) < 1e-7


@pytest.mark.parametrize('semilla', [4, 5])
def test_efectivo_no_negativo_y_comisiones_suman(semilla):
    datos = _historico_aleatorio(semilla)
    rng = np.random.default_rng(semilla)

    def estrategia(fecha, d):
        pesos = rng.dirichlet(np.ones(len(d)))
        pesos[rng.integers(len(d))] = 0.0  # siempre deja un activo a cero (fuerza cierres)
        return dict(zip(d, pesos / pesos.sum(), strict=True))

    cfg = BacktestConfig(start_date='2021-01-04', end_date='2022-12-30', initial_capital=50_000,
                         rebalance_frequency='monthly', monthly_contribution=300)
    e = BacktestEngine(cfg)
    r = e.run_backtest(estrategia, datos)
    assert (r.equity_curve['cash'] >= -1e-6).all()
    assert r.total_fees == pytest.approx(sum(t.commission for t in e.trades))
    assert r.total_contributions == pytest.approx(300 * len(pd.date_range('2021-01-04', '2022-12-30', freq='MS')))
    # Los cierres se valoran contra el precio medio previo, no contra el de salida
    cierres = [t for t in r.trades if t.side == 'sell']
    assert cierres and all(t.entry_price != t.exit_price for t in cierres)


def test_motor_reutilizable_da_el_mismo_resultado():
    datos = _historico_aleatorio(9)
    cfg = BacktestConfig(start_date='2021-01-04', end_date='2022-12-30', initial_capital=10_000,
                         rebalance_frequency='quarterly')
    e = BacktestEngine(cfg)
    r1 = e.run_backtest(lambda f, d: {t: 0.25 for t in d}, datos)
    r2 = e.run_backtest(lambda f, d: {t: 0.25 for t in d}, datos)
    assert r1.final_capital == pytest.approx(r2.final_capital)
    assert len(r1.equity_curve) == len(r2.equity_curve)


def test_alias_en_espanol_de_la_api_020():
    """scripts/run_fase2.py construye la config y lee el resultado con los nombres en español."""
    datos = _historico_aleatorio(11)
    cfg_es = BacktestConfig(fecha_inicio='2021-01-04', fecha_fin='2022-12-30', capital_inicial=10_000,
                            frecuencia_rebalanceo='monthly', commission_rate=0.001, slippage_pct=0.001)
    cfg_en = BacktestConfig(start_date='2021-01-04', end_date='2022-12-30', initial_capital=10_000,
                            rebalance_frequency='monthly')
    assert cfg_es == cfg_en
    assert cfg_es.fecha_inicio == '2021-01-04' and cfg_es.capital_inicial == 10_000
    r = BacktestEngine(cfg_es).run_backtest(lambda f, d: {t: 0.25 for t in d}, datos)
    assert r.retorno_total_pct == r.total_return_pct
    assert r.capital_inicial == 10_000
    assert r.trades_ganadores + r.trades_perdedores == len(r.trades)
    assert 'Retorno Total' in format_backtest_result(r)


def test_config_incompleta_o_divisa_sin_servicio():
    with pytest.raises(TypeError):
        BacktestConfig(start_date='2021-01-04')
    with pytest.raises(ValueError):
        BacktestConfig('2021-01-04', '2022-01-04', 1000, 'yearly')
    with pytest.raises(ValueError, match='servicio_cambio'):
        BacktestEngine(BacktestConfig('2021-01-04', '2022-01-04', 1000, 'monthly', user_currency='EUR'))


def test_servicio_con_benchmark_y_gancho_de_persistencia(cargador):
    guardados = []

    class ServicioConBD(PortfolioBacktestService):
        def _guardar_resultado(self, **kwargs):
            guardados.append(kwargs['nombre'])
            return 42

    servicio = ServicioConBD(cargador_precios=cargador)
    cfg = BacktestConfig(start_date=INICIO, end_date=FIN, initial_capital=10_000,
                         rebalance_frequency='quarterly', benchmark_ticker='BBB')
    res = servicio.run_backtest_with_benchmark(
        lambda f, d: {'AAA': 0.5, 'BBB': 0.5}, list(CRECIMIENTO), cfg, portfolio_id=1, cliente_id=2
    )
    assert res['testeo_id'] == 42 and guardados
    assert res['benchmark']['total_return_pct'] == pytest.approx(0.5, abs=0.01)
    assert bool(res['comparison']['outperformed'])
    assert set(res['config']) >= {'start_date', 'initial_capital', 'user_currency'}
    assert PortfolioBacktestService._convert_numpy_types({'a': np.float64(1.5)}) == {'a': 1.5}


# ---------------------------------------------------------------------------
# Rebalanceo en días hábiles y rotación real (2026-11-19)
# ---------------------------------------------------------------------------

def _contar_llamadas(cargador, frecuencia, inicio='2023-01-02', fin='2025-12-31'):
    """Ejecuta pesos iguales y devuelve (fechas en que la estrategia fue llamada, resultado)."""
    llamadas = []

    def estrategia(fecha, datos):
        llamadas.append(pd.Timestamp(fecha))
        return {'AAA': 0.5, 'BBB': 0.5}

    cfg = BacktestConfig(start_date=inicio, end_date=fin, initial_capital=10000,
                         rebalance_frequency=frecuencia, commission_rate=0.0,
                         slippage_pct=0.0, user_currency='USD')
    resultado = BacktestEngine(cfg, cargador_precios=cargador).run_backtest(estrategia, list(CRECIMIENTO))
    return llamadas, resultado


def test_rebalanceo_mensual_no_se_salta_meses_que_empiezan_en_festivo(cargador):
    # 36 meses: 1 asignación inicial (2023-01-02) + 35 rebalanceos, aunque el día 1 caiga en fin de semana
    llamadas, _ = _contar_llamadas(cargador, 'monthly')
    assert len(llamadas) == 36
    meses = {(f.year, f.month) for f in llamadas}
    assert len(meses) == 36
    # cada rebalanceo es el primer día hábil del mes
    for f in llamadas:
        assert f == pd.bdate_range(f.replace(day=1), periods=1)[0]


def test_rebalanceo_trimestral_y_semanal_en_dias_habiles(cargador):
    llamadas, _ = _contar_llamadas(cargador, 'quarterly')
    assert len(llamadas) == 12
    llamadas, _ = _contar_llamadas(cargador, 'weekly')
    semanas = {f.isocalendar()[:2] for f in llamadas}
    assert len(llamadas) == len(semanas)  # como mucho uno por semana
    assert len(llamadas) >= 155


def test_alinear_a_dias_de_trading_toma_el_siguiente_y_deduplica():
    dias = list(pd.to_datetime(['2024-06-03', '2024-06-04', '2024-07-01', '2024-07-02']))
    programadas = list(pd.to_datetime(['2024-06-01', '2024-06-02', '2024-07-01', '2024-08-01']))
    assert BacktestEngine._alinear_a_dias_de_trading(programadas, dias) == [
        pd.Timestamp('2024-06-03'), pd.Timestamp('2024-07-01'),
    ]


def test_rotacion_de_pesos_iguales_refleja_la_deriva(cargador):
    # A dobla y B sube un 50 %: volver a 50/50 cada mes obliga a vender A y comprar B
    _, r = _contar_llamadas(cargador, 'monthly')
    assert len(r.rotaciones) == 36
    fecha0, rot0 = r.rotaciones[0]
    assert rot0 == pytest.approx(1.0, abs=0.01)  # compra inicial: todo el capital
    rotaciones = [t for _, t in r.rotaciones[1:]]
    assert all(0.0 < t < 0.02 for t in rotaciones)
