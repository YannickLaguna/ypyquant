"""
test_paper_paridad.py — Paridad del paper interno con el backtest (el test que protege el motor).

Responsabilidad:
    Avanzar una sesión en modo `cierre` día a día (o partida en dos procesos, guardando y recargando
    `estado.json` en medio) da la MISMA curva de equity, las mismas operaciones, los mismos pesos y las mismas
    rotaciones (tolerancia 1e-9) que `simular` del mismo rango. Se prueba con `pesos_iguales`, `momentum` y
    `cruce_medias`. Si alguien cambia el motor (GrowSave lo usa), este test lo detecta.

Se relaciona con:
    src/ypyquant/paper/{simulador,almacen,sesion}.py y src/ypyquant/estrategias/simulacion.py (`simular`).
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, cargar_datos_experimento
from ypyquant.estrategias.receta import Receta
from ypyquant.estrategias.simulacion import simular
from ypyquant.paper.almacen import AlmacenPaper
from ypyquant.paper.servicio import ServicioPaper

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
NY = ZoneInfo("America/New_York")
INICIO, FIN = "2021-01-04", "2021-07-30"  # ~150 sesiones

ESTRATEGIAS = [
    ("pesos_iguales", {}, "monthly"),
    ("momentum", {"top_k": 3, "meses_retorno": 3}, "monthly"),
    ("cruce_medias", {"rapida": 10, "lenta": 40}, "weekly"),
    ("momentum", {"top_k": 2, "meses_retorno": 2}, "daily"),
]


def _reloj(dia: str):
    """Reloj fijo: las 20:00 de Nueva York del día dado (esa sesión ya está cerrada)."""
    return lambda: datetime.fromisoformat(f"{dia}T20:00:00").replace(tzinfo=NY)


def _receta(estrategia, parametros, frecuencia) -> Receta:
    return Receta(estrategia=estrategia, parametros=parametros, universo=TICKERS, frecuencia_rebalanceo=frecuencia,
                  comision=0.001, slippage=0.0005, dias_historia_previa=400).normalizada()


def _servicio(tmp_path, dia_actual: str) -> ServicioPaper:
    return ServicioPaper(AlmacenPaper(tmp_path / "paper"), FuenteSintetica(), ahora=_reloj(dia_actual))


def _backtest(receta: Receta, fin: str):
    config = ConfiguracionExperimento.desde_receta(receta, INICIO, fin, 100_000.0)
    return simular(config, cargar_datos_experimento(config, FuenteSintetica()), completo=False)


def _comparar(sesion_tablas, backtest, n_dias: int | None = None):
    equity = sesion_tablas.equity
    esperado = backtest.equity
    assert len(equity) == len(esperado)
    np.testing.assert_array_equal(pd.to_datetime(equity["fecha"]).values, pd.to_datetime(esperado["fecha"]).values)
    np.testing.assert_allclose(equity["equity"].to_numpy(float), esperado["equity"].to_numpy(float), rtol=1e-9, atol=1e-9)
    ops, ops_bt = sesion_tablas.operaciones, backtest.operaciones
    assert len(ops) == len(ops_bt)
    assert list(ops["ticker"]) == list(ops_bt["ticker"]) and list(ops["lado"]) == list(ops_bt["lado"])
    assert list(pd.to_datetime(ops["fecha"])) == list(pd.to_datetime(ops_bt["fecha"]))
    for columna in ("cantidad", "precio", "comision", "pnl"):
        np.testing.assert_allclose(ops[columna].to_numpy(float), ops_bt[columna].to_numpy(float), rtol=1e-9, atol=1e-9,
                                   equal_nan=True)
    pesos, pesos_bt = sesion_tablas.pesos, backtest.pesos
    assert list(pesos["ticker"]) == list(pesos_bt["ticker"])
    np.testing.assert_allclose(pesos["peso"].to_numpy(float), pesos_bt["peso"].to_numpy(float), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(sesion_tablas.rotaciones["rotacion"].to_numpy(float),
                               backtest.rotaciones["rotacion"].to_numpy(float), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("estrategia,parametros,frecuencia", ESTRATEGIAS)
def test_avanzar_dia_a_dia_equivale_al_backtest(tmp_path, estrategia, parametros, frecuencia):
    receta = _receta(estrategia, parametros, frecuencia)
    servicio = _servicio(tmp_path, FIN)
    sesion = servicio.crear_sesion(receta, nombre="paridad", fecha_alta=INICIO, modo_ejecucion="cierre")
    dias = pd.bdate_range(INICIO, FIN)
    for dia in dias:  # un avance por día: lo que haría el programador cada tarde
        servicio.simulador.avanzar(sesion.id, hasta=dia.date())
    tablas = servicio.almacen.leer_tablas(sesion.id)
    assert len(tablas.equity) == len(dias)
    _comparar(tablas, _backtest(receta, FIN))


@pytest.mark.parametrize("estrategia,parametros,frecuencia", ESTRATEGIAS[:3])
def test_catchup_en_bloque_y_partido_en_dos_procesos_dan_lo_mismo(tmp_path, estrategia, parametros, frecuencia):
    receta = _receta(estrategia, parametros, frecuencia)
    # (a) todo de golpe, como tras un PC apagado
    en_bloque = _servicio(tmp_path / "a", FIN)
    sesion_a = en_bloque.crear_sesion(receta, nombre="bloque", fecha_alta=INICIO, modo_ejecucion="cierre")
    en_bloque.simulador.avanzar(sesion_a.id)
    # (b) partido: el primer «proceso» llega a abril, el segundo (servicio nuevo, estado recargado de disco) acaba
    primero = _servicio(tmp_path / "b", FIN)
    sesion_b = primero.crear_sesion(receta, nombre="partido", fecha_alta=INICIO, modo_ejecucion="cierre")
    primero.simulador.avanzar(sesion_b.id, hasta=date(2021, 4, 30))
    segundo = ServicioPaper(AlmacenPaper(tmp_path / "b" / "paper"), FuenteSintetica(), ahora=_reloj(FIN))
    segundo.simulador.avanzar(sesion_b.id)
    ref = _backtest(receta, FIN)
    _comparar(en_bloque.almacen.leer_tablas(sesion_a.id), ref)
    _comparar(segundo.almacen.leer_tablas(sesion_b.id), ref)
