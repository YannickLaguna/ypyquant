"""
test_paper_paridad.py — Paridad del paper interno con el backtest (el test que protege el motor).

Responsabilidad:
    Avanzar una sesión en modo `cierre` día a día (o partida en dos procesos, guardando y recargando
    `estado.json` en medio) da la MISMA curva de equity, las mismas operaciones, los mismos pesos y las mismas
    rotaciones (tolerancia 1e-9) que `simular` del mismo rango. Se prueba con `pesos_iguales`, `momentum` y
    `cruce_medias`, y con margen: una receta apalancada (`bayes_stein` con volatilidad objetivo y L = 2), largo-corto
    (130/30 y dólar-neutral 100/100, con comisión, slippage y préstamo de títulos) y un caso con LLAMADA DE MARGEN
    (el corto se dispara: liquidación forzosa idéntica en el motor y en el paper, con su evento). También estrategias de señales
    largo-corto con stops, trailing, take profit y giro (`cruce_medias`, `canal_rsi_volatilidad`), con la sesión partida en dos
    procesos. Si alguien cambia el motor (GrowSave lo usa), este test lo detecta.

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


# ---- margen y cortos: la misma implementación en el motor y en el paper -----------------------------

ESTRATEGIAS_MARGEN = [
    ("momentum", {"top_k": 2, "meses_retorno": 3, "exposicion_larga": 1.3, "exposicion_corta": 0.3}, "monthly"),
    ("momentum", {"top_k": 2, "meses_retorno": 2, "exposicion_larga": 1.0, "exposicion_corta": 1.0,
                  "coste_prestamo_titulos": 0.02}, "weekly"),
    ("bayes_stein", {"ventana_dias": 250, "volatilidad_objetivo": 0.6, "apalancamiento_maximo": 2.0,
                     "diferencial_financiacion": 0.02, "tasa_libre_riesgo": 0.03}, "monthly"),
]


@pytest.mark.parametrize("estrategia,parametros,frecuencia", ESTRATEGIAS_MARGEN)
def test_paridad_con_margen_y_cortos(tmp_path, estrategia, parametros, frecuencia):
    receta = _receta(estrategia, parametros, frecuencia)
    servicio = _servicio(tmp_path, FIN)
    sesion = servicio.crear_sesion(receta, nombre="paridad margen", fecha_alta=INICIO, modo_ejecucion="cierre")
    for dia in pd.bdate_range(INICIO, FIN):
        servicio.simulador.avanzar(sesion.id, hasta=dia.date())
    referencia = _backtest(receta, FIN)
    tablas = servicio.almacen.leer_tablas(sesion.id)
    _comparar(tablas, referencia)
    if estrategia == "momentum":
        assert (tablas.pesos["peso"] < 0).any() and (tablas.operaciones["lado"] == "venta").any()
    else:
        assert any("Apalancamiento" in a for a in referencia.avisos)  # el préstamo existe y el paper lo cobró igual


def _fuente_con_corto_que_se_dispara():
    """AAA sube, CCC cae (se vende en corto) y a partir de febrero CCC se multiplica: el corto revienta el margen."""
    from tests._paper_utiles import FuenteManual, barras

    series = {t: barras("2019-01-01", FIN, base, deriva) for t, base, deriva in
              (("AAA", 50.0, 0.002), ("BBB", 60.0, 0.0), ("CCC", 70.0, -0.002), ("SPY", 100.0, 0.001))}
    ccc = series["CCC"]
    posicion = ccc.index.searchsorted(pd.Timestamp("2021-02-01"))
    factor = np.ones(len(ccc))
    factor[posicion:] = 1.10 ** np.arange(1, len(ccc) - posicion + 1)
    for columna in ("Open", "High", "Low", "Close", "adjusted_close"):
        ccc[columna] = ccc[columna].to_numpy() * factor
    return FuenteManual(series)


def test_llamada_de_margen_identica_en_motor_y_paper(tmp_path):
    fuente = _fuente_con_corto_que_se_dispara()
    receta = Receta(estrategia="momentum", parametros={"top_k": 1, "meses_retorno": 2, "exposicion_larga": 1.0,
                                                       "exposicion_corta": 1.0},
                    universo=["AAA", "BBB", "CCC"], frecuencia_rebalanceo="monthly", comision=0.001, slippage=0.0005,
                    dias_historia_previa=400).normalizada()
    servicio = ServicioPaper(AlmacenPaper(tmp_path / "paper"), fuente, ahora=_reloj(FIN))
    sesion = servicio.crear_sesion(receta, nombre="llamada", fecha_alta=INICIO, modo_ejecucion="cierre")
    for dia in pd.bdate_range(INICIO, FIN):
        servicio.simulador.avanzar(sesion.id, hasta=dia.date())
    config = ConfiguracionExperimento.desde_receta(receta, INICIO, FIN, 100_000.0)
    referencia = simular(config, cargar_datos_experimento(config, fuente), completo=False)
    tablas = servicio.almacen.leer_tablas(sesion.id)
    llamadas_bt = [a for a in referencia.avisos if a.startswith("Llamada de margen")]
    llamadas_paper = [e["mensaje"] for e in tablas.eventos if e["tipo"] == "llamada_margen"]
    assert llamadas_bt, "el corto se multiplica: debe haber al menos una llamada de margen"
    assert llamadas_paper == llamadas_bt
    _comparar(tablas, referencia)


# ---- estrategia de señales largo-corto con stops: la máquina de estados es la misma en el motor y en el paper ----------

SENALES_LARGO_CORTO = [
    ("cruce_medias", {"rapida": 5, "lenta": 20, "lados": "largo_corto", "exposicion_bruta_max": 1.5, "stop_perdida": 0.03,
                      "stop_trailing": 0.05, "take_profit": 0.08, "max_posiciones": 4, "peso_maximo": 0.4,
                      "permitir_giro": True, "coste_prestamo_titulos": 0.02}, "daily"),
    ("canal_rsi_volatilidad", {"ventana_canal": 15, "ventana_salida": 6, "ventana_rsi": 7, "lados": "largo_corto",
                               "exposicion_bruta_max": 1.2, "max_posiciones": 3, "max_dias_posicion": 25,
                               "stop_perdida": 0.04}, "weekly"),
]


@pytest.mark.parametrize("estrategia,parametros,frecuencia", SENALES_LARGO_CORTO)
def test_paridad_de_una_estrategia_de_senales_largo_corto_con_stops(tmp_path, estrategia, parametros, frecuencia):
    receta = _receta(estrategia, parametros, frecuencia)
    servicio = _servicio(tmp_path, FIN)
    sesion = servicio.crear_sesion(receta, nombre="paridad senales", fecha_alta=INICIO, modo_ejecucion="cierre")
    for dia in pd.bdate_range(INICIO, FIN):
        servicio.simulador.avanzar(sesion.id, hasta=dia.date())
    referencia = _backtest(receta, FIN)
    tablas = servicio.almacen.leer_tablas(sesion.id)
    _comparar(tablas, referencia)
    assert (tablas.pesos["peso"] < 0).any(), "el caso debe abrir cortos"
    assert {"entrada_corta"} <= set(referencia.operaciones["motivo"]) and {"stop_loss", "trailing", "take_profit", "senal_contraria"} & set(
        referencia.operaciones["motivo"]), "el caso debe ejercitar los stops y el giro"


def test_senales_largo_corto_partidas_en_dos_procesos_recargan_el_estado_con_los_cortos(tmp_path):
    estrategia, parametros, frecuencia = SENALES_LARGO_CORTO[0]
    receta = _receta(estrategia, parametros, frecuencia)
    primero = _servicio(tmp_path, FIN)
    sesion = primero.crear_sesion(receta, nombre="partido", fecha_alta=INICIO, modo_ejecucion="cierre")
    primero.simulador.avanzar(sesion.id, hasta=date(2021, 4, 30))
    segundo = ServicioPaper(AlmacenPaper(tmp_path / "paper"), FuenteSintetica(), ahora=_reloj(FIN))
    segundo.simulador.avanzar(sesion.id)
    _comparar(segundo.almacen.leer_tablas(sesion.id), _backtest(receta, FIN))
