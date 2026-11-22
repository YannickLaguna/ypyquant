"""
test_ejecutor_orquestador.py — El orquestador del bot (alpaca_vivo) con el broker SIMULADO, sin Alpaca.

Responsabilidad:
    Probar el cableado calendario -> estrategia -> Ejecutor -> broker con el adaptador en memoria:
    rebalanceo solo cuando toca, órdenes bloqueadas por órdenes abiertas, mercado cerrado, fills
    diferidos registrados una sola vez, y que tras ejecutar el plan la cuenta queda en los pesos. Es lo
    que demuestra que el puerto `Broker` es una costura real (dos adaptadores intercambiables).
    Complementa a tests/test_paper_vivo.py, que prueba el mismo bucle con un cliente de Alpaca falso.

Se relaciona con:
    - pipelines/paper_trading/alpaca_vivo.py y estado_bot.py (código bajo prueba).
    - src/ypyquant/execution/broker_simulado.py (el adaptador) y ejecutor.py.
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pipelines.paper_trading import alpaca_vivo
from pipelines.paper_trading.estado_bot import EstadoBot
from ypyquant.estrategias import obtener_estrategia
from ypyquant.execution.broker_simulado import BrokerSimulado
from ypyquant.execution.nautilus.registro_fills import RegistroFills


def _precios(n=30, valores=None):
    valores = valores or {"AAA": 100.0, "BBB": 50.0}
    fechas = pd.bdate_range(end="2026-11-19", periods=n)
    return pd.DataFrame({t: np.full(n, p) for t, p in valores.items()}, index=fechas)


@pytest.fixture
def estado(tmp_path):
    return EstadoBot.cargar(tmp_path / "estado.json", "pesos_iguales|{}|AAA,BBB")


@pytest.fixture
def estrategia():
    return obtener_estrategia("pesos_iguales")({})


def _rebalancear(broker, estrategia, estado, frecuencia, hoy, abierto=None):
    return alpaca_vivo.rebalancear_con_estrategia_registro(
        broker, estrategia, _precios(), broker.mercado_abierto() if abierto is None else abierto,
        estado, frecuencia, hoy=hoy)


def test_rebalancea_una_vez_por_mes_y_deja_la_cuenta_en_los_pesos(estrategia, estado):
    broker = BrokerSimulado(100_000.0, precios={"AAA": 100.0, "BBB": 50.0})
    assert _rebalancear(broker, estrategia, estado, "monthly", date(2026, 11, 2)) is not None
    assert broker.estado_cuenta().posiciones == {"AAA": 500, "BBB": 1000}
    assert estado.ultimo_rebalanceo == date(2026, 11, 2) and len(estado.ordenes_pendientes) == 2
    enviadas = len(broker.enviadas)
    assert _rebalancear(broker, estrategia, estado, "monthly", date(2026, 11, 20)) is None  # no toca
    assert _rebalancear(broker, estrategia, estado, "monthly", date(2026, 12, 1)) is not None
    assert len(broker.enviadas) == enviadas  # tocaba, pero la cuenta ya está en objetivo: sin órdenes


def test_ticker_con_orden_abierta_no_recibe_otra_y_el_resto_si(estrategia, estado):
    broker = BrokerSimulado(100_000.0, precios={"AAA": 100.0, "BBB": 50.0}, ejecucion_inmediata=False)
    broker.enviar_orden("AAA", "compra", 1)  # orden previa aún abierta sobre AAA
    _rebalancear(broker, estrategia, estado, "daily", date(2026, 11, 20))
    assert [t for t, _, _ in broker.enviadas[1:]] == ["BBB"]


def test_mercado_cerrado_calcula_pesos_pero_no_envia_ni_marca(estrategia, estado):
    broker = BrokerSimulado(100_000.0, precios={"AAA": 100.0, "BBB": 50.0}, mercado_abierto=False)
    assert _rebalancear(broker, estrategia, estado, "daily", date(2026, 11, 20)) is not None
    assert broker.enviadas == [] and estado.ultimo_rebalanceo is None


def test_si_no_se_pueden_consultar_las_abiertas_no_envia_nada(estrategia, estado):
    broker = BrokerSimulado(100_000.0, precios={"AAA": 100.0, "BBB": 50.0})
    broker.fallar_consulta_abiertas = True
    _rebalancear(broker, estrategia, estado, "daily", date(2026, 11, 20))
    assert broker.enviadas == [] and estado.ultimo_rebalanceo is None


def test_fills_diferidos_se_registran_una_sola_vez(estrategia, estado, tmp_path):
    broker = BrokerSimulado(100_000.0, precios={"AAA": 100.0, "BBB": 50.0}, ejecucion_inmediata=False)
    _rebalancear(broker, estrategia, estado, "daily", date(2026, 11, 20))
    registro = RegistroFills(tmp_path / "fills.jsonl")
    assert alpaca_vivo.sincronizar_fills(broker, registro, estado, reintentos=0, espera_s=0) == 0  # aún abiertas
    broker.ejecutar_pendientes()
    assert alpaca_vivo.sincronizar_fills(broker, registro, estado, reintentos=0, espera_s=0) == 2
    assert alpaca_vivo.sincronizar_fills(broker, registro, estado, reintentos=0, espera_s=0) == 0
    filas = [json.loads(x) for x in (tmp_path / "fills.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {f["ticker"]: f["cantidad"] for f in filas} == {"AAA": 500, "BBB": 1000}
    assert estado.ordenes_pendientes == []


def test_modo_senales_usa_el_precio_del_broker_y_decide_por_senal():
    from types import SimpleNamespace

    broker = BrokerSimulado(100_000.0, precios={"AAA": 100.0})
    estrategia = SimpleNamespace(umbral_senal=0.3, config_riesgo={"max_position_pct": 0.10})
    id_orden = alpaca_vivo.evaluar_y_enviar(broker, "AAA", 1.0, estrategia, 100_000.0, 0, True)
    assert id_orden is not None and broker.estado_cuenta().posiciones == {"AAA": 100}  # 10 % de 100 000 / 100
    assert alpaca_vivo.evaluar_y_enviar(broker, "AAA", 1.0, estrategia, 100_000.0, 100, True) is None  # ya en objetivo
    assert alpaca_vivo.evaluar_y_enviar(broker, "AAA", 1.0, estrategia, 100_000.0, 0, False) is None  # mercado cerrado
    alpaca_vivo.evaluar_y_enviar(broker, "AAA", -1.0, estrategia, 100_000.0, 100, True)
    assert broker.estado_cuenta().posiciones == {}
