"""
test_margen_y_cortos.py — Reglas de margen compartidas, cortos en el Ejecutor, el broker simulado y el motor.

Responsabilidad:
    Protege `ypyquant.execution.margen` (requisito inicial, mantenimiento, factor de liquidación, cargos), los cortos
    del Ejecutor (abrir, cubrir, cruzar por cero en dos órdenes; sin la política de cortos todo igual que antes), el
    `BrokerSimulado` con margen y el `BacktestEngine` (valoración de cortos, préstamo de títulos, llamada de margen con
    liquidación proporcional y aviso). Con las reglas por defecto (L = 1, sin cortos) nada cambia.

Se relaciona con:
    src/ypyquant/execution/{margen,ejecutor,broker_simulado}.py y src/ypyquant/portfolio/backtest_engine.py;
    tests/test_paper_paridad.py (paridad paper-backtest con margen) y tests/test_ejecutor.py.
"""

import numpy as np
import pandas as pd
import pytest

from ypyquant.execution.broker_simulado import BrokerSimulado
from ypyquant.execution.ejecutor import (
    COMPRA,
    POLITICA_BOT,
    POLITICA_MOTOR,
    VENTA,
    EstadoCuenta,
    PoliticaEjecucion,
    planificar_rebalanceo,
)
from ypyquant.execution.margen import (
    ReglasMargen,
    cargos_diarios,
    capacidad_de_compra,
    evaluar_margen,
    factor_de_conservacion,
    reglas_desde_parametros,
    valores_largo_corto,
)
from ypyquant.portfolio.backtest_engine import BacktestConfig, BacktestEngine

CORTOS = PoliticaEjecucion(fraccionable=True, umbral_absoluto=10.0, permite_cortos=True)
CORTOS_ENTEROS = PoliticaEjecucion(fraccionable=False, umbral_relativo=0.01, permite_cortos=True)


# ---- reglas puras -----------------------------------------------------------------------------------

def test_reglas_por_defecto_estan_inactivas_y_validan():
    reglas = ReglasMargen()
    assert not reglas.activo and reglas.limite_bruta == 1.0
    assert ReglasMargen(apalancamiento_maximo=2.0).limite_bruta == 2.0
    assert ReglasMargen(apalancamiento_maximo=2.0, requisito_inicial=0.6).limite_bruta == pytest.approx(1 / 0.6)
    with pytest.raises(ValueError):
        ReglasMargen(apalancamiento_maximo=0.9)
    with pytest.raises(ValueError):
        ReglasMargen(mantenimiento_largo=0.6)


def test_evaluar_margen_detecta_la_llamada_por_debajo_del_mantenimiento():
    reglas = ReglasMargen(apalancamiento_maximo=2.0, permite_cortos=True)
    sano = evaluar_margen(100.0, 150.0, 30.0, reglas)  # mantenimiento = .25x150 + .30x30 = 46,5
    assert sano.requisito_mantenimiento == pytest.approx(46.5) and not sano.en_llamada and sano.neta == 120.0
    assert evaluar_margen(46.0, 150.0, 30.0, reglas).en_llamada
    assert not evaluar_margen(46.0, 150.0, 30.0, ReglasMargen()).en_llamada  # reglas inactivas: nunca hay llamada


def test_factor_de_conservacion_vuelve_al_requisito_inicial_descontando_costes():
    reglas = ReglasMargen(apalancamiento_maximo=2.0)
    assert factor_de_conservacion(30.0, 100.0, reglas) == pytest.approx(0.6)  # 30 = 0,5 x (0,6 x 100)
    f = factor_de_conservacion(30.0, 100.0, reglas, coste_unitario=0.01)
    equity_final = 30.0 - 0.01 * (1 - f) * 100.0
    assert equity_final == pytest.approx(0.5 * f * 100.0)  # tras pagar los costes se cumple justo el requisito
    assert factor_de_conservacion(60.0, 100.0, reglas) == 1.0  # ya cumple: no se reduce
    assert factor_de_conservacion(0.0, 100.0, reglas) == 0.0 and factor_de_conservacion(5.0, 0.0, reglas) == 1.0


def test_capacidad_de_compra_y_cargos():
    assert capacidad_de_compra(100.0, 100.0, 0.0, 0.0, ReglasMargen()) == 100.0
    assert capacidad_de_compra(-20.0, 100.0, 120.0, 0.0, ReglasMargen(apalancamiento_maximo=1.5)) == pytest.approx(30.0)
    con_cortos = ReglasMargen(apalancamiento_maximo=1.6, permite_cortos=True)
    assert capacidad_de_compra(130.0, 100.0, 0.0, 30.0, con_cortos) == pytest.approx(130.0)  # manda la bruta, no el efectivo
    reglas = ReglasMargen(apalancamiento_maximo=2.0, tasa_financiacion=0.0648, coste_prestamo_titulos=0.036, permite_cortos=True)
    interes, prestamo = cargos_diarios(-1000.0, 500.0, 3, reglas)
    assert interes == pytest.approx(1000 * 0.0648 * 3 / 360) and prestamo == pytest.approx(500 * 0.036 * 3 / 360)
    assert cargos_diarios(1000.0, 0.0, 3, reglas) == (0.0, 0.0)


def test_reglas_desde_parametros_largo_corto_y_apalancamiento():
    assert not reglas_desde_parametros({}).activo
    reglas = reglas_desde_parametros({"exposicion_larga": 1.3, "exposicion_corta": 0.3, "coste_prestamo_titulos": 0.004})
    assert reglas.permite_cortos and reglas.limite_bruta == pytest.approx(1.6) and reglas.coste_prestamo_titulos == 0.004
    apalancada = reglas_desde_parametros({"apalancamiento_maximo": 1.5, "diferencial_financiacion": 0.01, "tasa_libre_riesgo": 0.04})
    assert apalancada.tasa_financiacion == pytest.approx(0.05) and not apalancada.permite_cortos


# ---- Ejecutor con cortos ----------------------------------------------------------------------------

def test_sin_politica_de_cortos_un_peso_negativo_se_ignora_como_siempre():
    cuenta = EstadoCuenta(equity=1000.0, efectivo=1000.0)
    plan = planificar_rebalanceo({"A": 0.5, "B": -0.5}, {"A": 10.0, "B": 10.0}, cuenta, POLITICA_MOTOR)
    assert [o.ticker for o in plan.ordenes] == ["A"]
    assert [o.ticker for o in planificar_rebalanceo({"B": -0.5}, {"B": 10.0}, cuenta, POLITICA_BOT).ordenes] == []


def test_abre_cortos_con_la_politica_y_ordena_ventas_antes_que_compras():
    cuenta = EstadoCuenta(equity=1000.0, efectivo=1000.0)
    plan = planificar_rebalanceo({"L": 1.3, "C": -0.3}, {"L": 10.0, "C": 20.0}, cuenta, CORTOS)
    assert [(o.ticker, o.lado) for o in plan.ordenes] == [("C", VENTA), ("L", COMPRA)]  # el corto financia la compra
    corto = plan.ordenes[0]
    assert corto.importe == pytest.approx(300.0) and corto.cantidad == pytest.approx(15.0) and corto.cantidad_objetivo == pytest.approx(-15.0)


def test_cubre_un_corto_cuando_el_objetivo_es_cero_y_con_enteros_abre_con_floor_hacia_cero():
    cuenta = EstadoCuenta(equity=1000.0, efectivo=1300.0, posiciones={"C": -15.0})
    plan = planificar_rebalanceo({"L": 0.5}, {"L": 10.0, "C": 20.0}, cuenta, CORTOS)
    assert plan.ordenes[0].ticker == "C" and plan.ordenes[0].lado == COMPRA and plan.ordenes[0].motivo == "cierre"
    assert plan.ordenes[0].cantidad == pytest.approx(15.0)
    enteros = planificar_rebalanceo({"C": -0.33}, {"C": 20.0}, EstadoCuenta(1000.0, 1000.0), CORTOS_ENTEROS)
    assert enteros.ordenes[0].cantidad == 16.0 and enteros.ordenes[0].cantidad_objetivo == -16.0  # 330/20 = 16,5 -> 16


def test_cambio_de_signo_son_dos_ordenes_cierre_y_apertura():
    precios = {"X": 10.0}
    largo_a_corto = planificar_rebalanceo({"X": -0.5}, precios, EstadoCuenta(1000.0, 500.0, {"X": 50.0}), CORTOS)
    assert [(o.lado, o.motivo, round(o.cantidad, 6)) for o in largo_a_corto.ordenes] == [
        (VENTA, "cierre", 50.0), (VENTA, "rebalanceo", 50.0)]
    corto_a_largo = planificar_rebalanceo({"X": 0.5}, precios, EstadoCuenta(1000.0, 1500.0, {"X": -50.0}), CORTOS)
    assert [(o.lado, o.motivo, round(o.cantidad, 6)) for o in corto_a_largo.ordenes] == [
        (COMPRA, "cierre", 50.0), (COMPRA, "rebalanceo", 50.0)]


def test_con_cortos_no_se_normaliza_la_suma_de_pesos():
    # dólar-neutral 100/100: suma 0; con la política de cortos las exposiciones se respetan tal cual
    plan = planificar_rebalanceo({"L": 1.0, "C": -1.0}, {"L": 10.0, "C": 10.0}, EstadoCuenta(1000.0, 1000.0), CORTOS)
    assert {o.ticker: o.importe for o in plan.ordenes} == {"L": pytest.approx(1000.0), "C": pytest.approx(1000.0)}


# ---- BrokerSimulado con margen ----------------------------------------------------------------------

def test_broker_sin_reglas_rechaza_cortos_y_apalancamiento():
    broker = BrokerSimulado(1000.0, precios={"A": 10.0})
    assert broker.consultar_orden(broker.enviar_orden("A", VENTA, 5)).estado == "rejected"
    assert broker.consultar_orden(broker.enviar_orden("A", COMPRA, 200)).estado == "rejected"  # 2000 > 1000


def test_broker_con_margen_vende_en_corto_cubre_y_limita_la_exposicion():
    reglas = ReglasMargen(apalancamiento_maximo=1.6, permite_cortos=True)
    broker = BrokerSimulado(1000.0, precios={"A": 10.0, "B": 10.0}, reglas_margen=reglas)
    assert broker.consultar_orden(broker.enviar_orden("A", VENTA, 30)).estado == "filled"  # corto de 300
    assert broker.posiciones["A"] == -30 and broker.efectivo == pytest.approx(1300.0)
    assert broker.consultar_orden(broker.enviar_orden("B", COMPRA, 130)).estado == "filled"  # largo de 1300 -> bruta 1,6
    assert broker.consultar_orden(broker.enviar_orden("B", COMPRA, 1)).estado == "rejected"  # pasaría de 1,6x
    assert broker.consultar_orden(broker.enviar_orden("A", COMPRA, 31)).estado == "rejected"  # no cruza por cero
    assert broker.consultar_orden(broker.enviar_orden("A", COMPRA, 30)).estado == "filled"
    assert "A" not in broker.posiciones and broker.estado_cuenta().equity == pytest.approx(1000.0)


# ---- Motor ------------------------------------------------------------------------------------------

def _precios(**series) -> dict[str, pd.DataFrame]:
    indice = pd.bdate_range("2021-01-04", periods=len(next(iter(series.values()))))
    return {t: pd.DataFrame({"Close": pd.Series(v, index=indice, dtype=float),
                             "adjusted_close": pd.Series(v, index=indice, dtype=float)}) for t, v in series.items()}


def _motor(n_dias: int, **extra) -> BacktestEngine:
    fin = (pd.bdate_range("2021-01-04", periods=n_dias)[-1]).strftime("%Y-%m-%d")
    config = BacktestConfig(start_date="2021-01-04", end_date=fin, initial_capital=100_000.0,
                            rebalance_frequency="monthly", commission_rate=0.0, slippage_pct=0.0)
    return BacktestEngine(config, **extra)


def test_motor_largo_corto_gana_con_el_perdedor_y_paga_el_prestamo_de_titulos():
    n = 60
    precios = _precios(GAN=100 * 1.002 ** np.arange(n), PER=100 * 0.998 ** np.arange(n))
    reglas = ReglasMargen(apalancamiento_maximo=2.0, tasa_financiacion=0.05, coste_prestamo_titulos=0.01, permite_cortos=True)
    motor = _motor(n, reglas_margen=reglas)
    resultado = motor.run_backtest(lambda f, d: {"GAN": 1.0, "PER": -1.0}, precios)
    assert resultado.total_return_pct > 0.10  # gana el largo y gana el corto
    assert motor.positions["PER"]["quantity"] < 0 < motor.positions["GAN"]["quantity"]
    assert motor.coste_prestamo_titulos > 0 and motor.exposicion_maxima == pytest.approx(2.0, rel=1e-3)
    assert not motor.llamadas_margen
    solo_largo = _motor(n).run_backtest(lambda f, d: {"GAN": 1.0}, precios)
    assert resultado.total_return_pct > solo_largo.total_return_pct


def test_motor_cubrir_un_corto_anota_pnl_positivo_si_el_precio_cae():
    n = 40
    precios = _precios(A=100 * 0.99 ** np.arange(n), B=np.full(n, 100.0))
    reglas = ReglasMargen(apalancamiento_maximo=1.5, permite_cortos=True)
    motor = _motor(n, reglas_margen=reglas)
    pesos = iter([{"A": -0.5, "B": 0.5}] + [{"B": 0.5}] * 10)
    resultado = motor.run_backtest(lambda f, d: next(pesos), precios)
    cierres = [t for t in resultado.trades if t.ticker == "A"]
    assert cierres and cierres[0].side == "buy" and cierres[0].pnl > 0
    assert "A" not in motor.positions


def test_llamada_de_margen_liquida_proporcionalmente_y_queda_anotada():
    n = 30
    subida = np.concatenate([np.full(5, 100.0), 100 * 1.12 ** np.arange(1, n - 4)])  # el activo corto se dispara
    precios = _precios(L=np.full(n, 100.0), C=subida)
    reglas = ReglasMargen(apalancamiento_maximo=2.0, permite_cortos=True)
    motor = _motor(n, reglas_margen=reglas)
    resultado = motor.run_backtest(lambda f, d: {"L": 1.0, "C": -1.0}, precios)
    assert motor.llamadas_margen, "el corto se dispara: el equity cae bajo el mantenimiento"
    llamada = motor.llamadas_margen[0]
    assert llamada["equity"] < llamada["requisito_mantenimiento"] and 0.0 <= llamada["factor_conservado"] < 1.0
    # tras liquidar se vuelve al requisito inicial (50 %) y el motor no queda con exposición desproporcionada
    ultimo_valor = motor.portfolio_value
    assert ultimo_valor > 0 and resultado.max_drawdown_pct < 0


def test_motor_con_reglas_por_defecto_no_cambia():
    n = 60
    precios = _precios(A=100 * 1.001 ** np.arange(n), B=100 * 1.0005 ** np.arange(n))
    estrategia = lambda f, d: {"A": 0.5, "B": 0.5}  # noqa: E731
    base = _motor(n).run_backtest(estrategia, precios)
    con_reglas = _motor(n, reglas_margen=ReglasMargen()).run_backtest(estrategia, precios)
    np.testing.assert_array_equal(base.equity_curve["equity"].to_numpy(), con_reglas.equity_curve["equity"].to_numpy())
    assert valores_largo_corto({"A": 1.0, "B": -2.0}, {"A": 10.0, "B": 5.0}) == (10.0, 10.0)


# ---- Estrategias largo-corto (momentum) -------------------------------------------------------------

def _datos_momentum(n_tickers=10, n=400):
    indice = pd.bdate_range("2020-01-02", periods=n)
    generador = np.random.default_rng(3)
    cierres = {f"T{i}": pd.Series(100 * np.exp(np.cumsum(generador.normal(0.0004 * (i - 5), 0.01, n))), index=indice)
               for i in range(n_tickers)}
    return {t: pd.DataFrame({"Close": c, "adjusted_close": c}) for t, c in cierres.items()}


def test_momentum_largo_corto_130_30_y_dolar_neutral():
    from ypyquant.estrategias.registro import obtener_estrategia

    datos = _datos_momentum()
    fecha = next(iter(datos.values())).index[-1]
    e = obtener_estrategia("momentum")({"top_k": 3, "meses_retorno": 6, "exposicion_larga": 1.3, "exposicion_corta": 0.3})
    assert e.admite_cortos()
    pesos = e.como_funcion()(fecha, datos)
    largos = {t: w for t, w in pesos.items() if w > 0}
    cortos = {t: w for t, w in pesos.items() if w < 0}
    assert len(largos) == 3 and len(cortos) == 3 and not set(largos) & set(cortos)
    assert sum(largos.values()) == pytest.approx(1.3) and sum(cortos.values()) == pytest.approx(-0.3)
    neutral = obtener_estrategia("momentum")({"top_k": 2, "meses_retorno": 6, "exposicion_larga": 1.0, "exposicion_corta": 1.0})
    pesos = neutral.como_funcion()(fecha, datos)
    assert sum(pesos.values()) == pytest.approx(0.0, abs=1e-12) and sum(abs(w) for w in pesos.values()) == pytest.approx(2.0)


def test_momentum_sin_cortos_no_cambia_y_valida_las_exposiciones():
    from ypyquant.estrategias.registro import obtener_estrategia

    datos = _datos_momentum()
    fecha = next(iter(datos.values())).index[-1]
    e = obtener_estrategia("momentum")({"top_k": 3, "meses_retorno": 6})
    assert not e.admite_cortos() and all(w > 0 for w in e.como_funcion()(fecha, datos).values())
    with pytest.raises(ValueError, match="exposicion_larga"):
        obtener_estrategia("momentum")({"exposicion_larga": 1.3})
    with pytest.raises(ValueError, match="2x"):
        obtener_estrategia("momentum")({"exposicion_larga": 1.6, "exposicion_corta": 0.6})


def test_parametros_de_cortos_fuera_de_huella_en_su_defecto():
    from ypyquant.estrategias.experimentos import ConfiguracionExperimento

    def huella(**extra):
        return ConfiguracionExperimento(estrategia="momentum", universo=["A", "B"], fecha_inicio="2021-01-04",
                                        fecha_fin="2021-06-30", parametros={"top_k": 2, **extra}).huella()

    assert huella() == huella(exposicion_larga=1.0, exposicion_corta=0.0, top_k_cortos=None, coste_prestamo_titulos=0.003)
    assert huella() != huella(exposicion_larga=1.3, exposicion_corta=0.3)


def test_experimento_largo_corto_de_extremo_a_extremo_avisa_y_opera_cortos():
    from ypyquant.data.fuentes import FuenteSintetica
    from ypyquant.estrategias.experimentos import ConfiguracionExperimento, cargar_datos_experimento
    from ypyquant.estrategias.simulacion import simular

    config = ConfiguracionExperimento(
        estrategia="momentum", universo=["AAA", "BBB", "CCC", "DDD", "EEE"], fecha_inicio="2021-01-04",
        fecha_fin="2021-06-30", frecuencia_rebalanceo="monthly", dias_historia_previa=400,
        parametros={"top_k": 2, "meses_retorno": 3, "exposicion_larga": 1.3, "exposicion_corta": 0.3})
    simulacion = simular(config, cargar_datos_experimento(config, FuenteSintetica()), completo=False)
    assert (simulacion.pesos["peso"] < 0).any()
    assert any(a.startswith("Cortos: préstamo de títulos") for a in simulacion.avisos)
    ventas_sin_pnl = simulacion.operaciones[(simulacion.operaciones["lado"] == "venta") & simulacion.operaciones["pnl"].isna()]
    assert not ventas_sin_pnl.empty  # aperturas de corto


def test_metricas_de_exposicion_bruta_y_neta_solo_con_cortos_o_prestamo():
    from ypyquant.data.fuentes import FuenteSintetica
    from ypyquant.estrategias.evaluacion import evaluar
    from ypyquant.estrategias.experimentos import ConfiguracionExperimento, cargar_datos_experimento
    from ypyquant.estrategias.simulacion import simular

    def metricas(**parametros):
        config = ConfiguracionExperimento(
            estrategia="momentum", universo=["AAA", "BBB", "CCC", "DDD", "EEE"], fecha_inicio="2021-01-04",
            fecha_fin="2021-06-30", frecuencia_rebalanceo="monthly", dias_historia_previa=400,
            parametros={"top_k": 2, "meses_retorno": 3, **parametros})
        return evaluar(simular(config, cargar_datos_experimento(config, FuenteSintetica()), completo=False)).metricas

    largo_corto = metricas(exposicion_larga=1.3, exposicion_corta=0.3)
    assert largo_corto["exposicion_bruta_maxima"] == pytest.approx(1.6) and largo_corto["exposicion_neta_media"] == pytest.approx(1.0)
    assert largo_corto["exposicion_corta_media"] == pytest.approx(0.3) and largo_corto["llamadas_margen"] == 0
    solo_largo = metricas()
    assert not any(clave.startswith("exposicion") for clave in solo_largo) and "llamadas_margen" not in solo_largo
