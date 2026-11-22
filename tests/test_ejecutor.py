"""
test_ejecutor.py — Tests del Ejecutor (pesos -> órdenes) y del broker simulado.

Responsabilidad:
    Probar `planificar_rebalanceo` por sí solo (puro, sin motor ni bot): semántica de CASH, motivos de
    omisión, cierres, acciones enteras frente a fracciones, normalización y rotación; y el
    `BrokerSimulado` como adaptador del puerto `Broker` (ejecución con costes, rechazos, órdenes abiertas).
    Los casos de equivalencia con el comportamiento previo del motor y del bot están en
    tests/test_caracterizacion_ejecucion.py.

Se relaciona con:
    - src/ypyquant/execution/ejecutor.py, broker.py y broker_simulado.py (código bajo prueba).
    - tests/test_ejecutor_orquestador.py (el bucle del bot con el broker simulado).
"""

from dataclasses import replace

import pytest

from ypyquant.execution.broker import Broker
from ypyquant.execution.broker_simulado import BrokerSimulado
from ypyquant.execution.ejecutor import (
    POLITICA_BOT,
    POLITICA_MOTOR,
    EstadoCuenta,
    PoliticaEjecucion,
    planificar_rebalanceo,
)

CUENTA = EstadoCuenta(equity=10_000.0, efectivo=10_000.0)


def test_cash_es_efectivo_nunca_una_orden_ni_una_omision_por_precio():
    for politica in (POLITICA_MOTOR, POLITICA_BOT, PoliticaEjecucion()):
        plan = planificar_rebalanceo({"AAA": 0.5, "CASH": 0.5}, {"AAA": 100.0}, CUENTA, politica)
        assert [o.ticker for o in plan.ordenes] == ["AAA"]
        assert plan.ordenes[0].importe == pytest.approx(5000.0)
        assert plan.omitidas == ()  # CASH no aparece como «sin precio»


def test_rotacion_real_con_cash_y_rotacion_historica_del_motor():
    pesos, precios = {"AAA": 0.5, "CASH": 0.5}, {"AAA": 100.0}
    assert planificar_rebalanceo(pesos, precios, CUENTA, PoliticaEjecucion()).rotacion == pytest.approx(0.5)
    assert planificar_rebalanceo(pesos, precios, CUENTA, POLITICA_MOTOR).rotacion == pytest.approx(1.0)  # compat.


def test_posicion_en_cash_se_ignora_y_no_se_vende():
    cuenta = EstadoCuenta(10_000.0, 0.0, {"CASH": 5.0, "AAA": 100.0})
    plan = planificar_rebalanceo({"AAA": 1.0}, {"AAA": 100.0}, cuenta, POLITICA_BOT)
    assert plan.ordenes == ()


def test_omisiones_con_motivo():
    cuenta = EstadoCuenta(10_000.0, 0.0, {"BBB": 10.0, "CCC": 5.0, "DDD": 49.0})
    plan = planificar_rebalanceo(
        {"AAA": 0.2, "DDD": 0.5005, "EEE": 0.1}, {"AAA": 100.0, "BBB": 10.0, "DDD": 100.0},
        cuenta, replace(POLITICA_BOT, umbral_relativo=0.02), tickers_bloqueados={"BBB"})
    assert {(o.ticker, o.motivo) for o in plan.omitidas} == {
        ("BBB", "bloqueado"), ("CCC", "sin_precio"), ("DDD", "bajo_umbral"), ("EEE", "sin_precio")}
    assert [(o.ticker, o.lado) for o in plan.ordenes] == [("AAA", "compra")]


def test_cierre_lleva_motivo_y_ignora_el_umbral():
    plan = planificar_rebalanceo({"AAA": 1.0}, {"AAA": 100.0, "BBB": 1.0},
                                 EstadoCuenta(10_000.0, 0.0, {"AAA": 100.0, "BBB": 2.0}), POLITICA_BOT)
    assert [(o.ticker, o.lado, o.motivo) for o in plan.ordenes] == [("BBB", "venta", "cierre")]


def test_acciones_enteras_frente_a_fracciones():
    entera = planificar_rebalanceo({"AAA": 1.0}, {"AAA": 30.0}, CUENTA, POLITICA_BOT).ordenes[0]
    fraccion = planificar_rebalanceo({"AAA": 1.0}, {"AAA": 30.0}, CUENTA, PoliticaEjecucion()).ordenes[0]
    assert entera.cantidad == 333 and entera.importe == pytest.approx(333 * 30.0)
    assert fraccion.cantidad == pytest.approx(10_000.0 / 30.0) and fraccion.importe == pytest.approx(10_000.0)


def test_solo_la_politica_con_tolerancia_normaliza():
    pesos, precios = {"AAA": 0.25}, {"AAA": 100.0}
    assert planificar_rebalanceo(pesos, precios, CUENTA, POLITICA_MOTOR).ordenes[0].importe == pytest.approx(10_000.0)
    assert planificar_rebalanceo(pesos, precios, CUENTA, POLITICA_BOT).ordenes[0].importe == pytest.approx(2_500.0)


def test_nunca_abre_cortos_ni_vende_mas_de_lo_que_hay():
    plan = planificar_rebalanceo({"AAA": -0.3, "BBB": 0.0}, {"AAA": 10.0, "BBB": 10.0}, CUENTA, PoliticaEjecucion())
    assert plan.ordenes == ()
    cuenta = EstadoCuenta(1000.0, 0.0, {"AAA": 4.0})
    plan = planificar_rebalanceo({"AAA": -0.3}, {"AAA": 10.0}, cuenta, PoliticaEjecucion())
    assert [(o.lado, o.cantidad) for o in plan.ordenes] == [("venta", 4.0)]


# ---- BrokerSimulado ---------------------------------------------------------------------------


def test_broker_simulado_cumple_el_puerto_y_aplica_costes():
    broker = BrokerSimulado(10_000.0, comision=0.001, slippage=0.001, precios={"AAA": 100.0})
    assert isinstance(broker, Broker)
    id_compra = broker.enviar_orden("AAA", "compra", 10)
    orden = broker.consultar_orden(id_compra)
    assert orden.terminal and orden.estado == "filled" and orden.precio_medio == pytest.approx(100.1)
    assert broker.efectivo == pytest.approx(10_000.0 - 10 * 100.1 * 1.001)
    assert broker.estado_cuenta().posiciones == {"AAA": 10}
    broker.enviar_orden("AAA", "venta", 10)
    assert broker.estado_cuenta().posiciones == {} and broker.efectivo < 10_000.0  # pagó costes de ida y vuelta


def test_broker_simulado_rechaza_lo_inviable():
    broker = BrokerSimulado(1000.0, precios={"AAA": 100.0})
    assert broker.consultar_orden(broker.enviar_orden("AAA", "compra", 11)).estado == "rejected"  # sin efectivo
    assert broker.consultar_orden(broker.enviar_orden("AAA", "venta", 1)).estado == "rejected"  # sin posición
    assert broker.consultar_orden(broker.enviar_orden("ZZZ", "compra", 1)).estado == "rejected"  # sin precio
    assert broker.efectivo == 1000.0 and broker.posiciones == {}


def test_broker_simulado_ordenes_abiertas_y_ejecucion_diferida():
    broker = BrokerSimulado(10_000.0, precios={"AAA": 100.0}, ejecucion_inmediata=False)
    id_orden = broker.enviar_orden("AAA", "compra", 5)
    assert broker.tickers_con_orden_abierta() == {"AAA"} and not broker.consultar_orden(id_orden).terminal
    broker.ejecutar_pendientes()
    assert broker.tickers_con_orden_abierta() == set() and broker.consultar_orden(id_orden).cantidad_ejecutada == 5
    broker.fallar_consulta_abiertas = True
    with pytest.raises(ConnectionError):
        broker.tickers_con_orden_abierta()


def test_el_plan_del_ejecutor_se_ejecuta_en_el_simulado_y_converge_a_los_pesos():
    broker = BrokerSimulado(100_000.0, precios={"AAA": 100.0, "BBB": 40.0})
    plan = planificar_rebalanceo({"AAA": 0.6, "BBB": 0.4}, broker.precios, broker.estado_cuenta(), POLITICA_BOT)
    for orden in plan.ordenes:
        broker.enviar_orden(orden.ticker, orden.lado, int(orden.cantidad))
    cuenta = broker.estado_cuenta()
    assert cuenta.posiciones == {"AAA": 600, "BBB": 1000}
    # Ya en objetivo: un segundo plan no propone nada (idempotente)
    assert planificar_rebalanceo({"AAA": 0.6, "BBB": 0.4}, broker.precios, cuenta, POLITICA_BOT).ordenes == ()
