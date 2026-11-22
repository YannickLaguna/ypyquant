"""
test_paper_simulador.py — Comportamiento del simulador del paper interno con barras fabricadas.

Responsabilidad:
    - 3.1 calendario: sesiones cerradas, pendientes y rebalanceo coherente con el calendario único.
    - 3.4 modo `apertura_siguiente`: decisión en D, ejecución a la apertura de D+1 con slippage; la última decisión
      queda pendiente.
    - 3.5 re-base por ajuste: un dividendo (el histórico ajustado se reescala) no produce salto en la equity.
    - 3.6 almacén y reanudación: una caída entre las tablas y `estado.json` se repite sin duplicar filas; bloqueos.
    - 3.7 día en curso, barras parciales y tickers que desaparecen.
    - 3.8 varias sesiones, una sola descarga por ticker.
    - 3.9 el estado de cada estrategia registrada es JSON.
    - Las métricas forward son las del backtest (`evaluar`).

Se relaciona con: src/ypyquant/paper/*.py y tests/_paper_utiles.py.
"""

import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests._paper_utiles import FuenteManual, barras, reloj
from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, cargar_datos_experimento, ejecutar_con_datos
from ypyquant.estrategias.receta import Receta
from ypyquant.estrategias.registro import listar_estrategias, obtener_estrategia
from ypyquant.execution.calendario import fechas_de_rebalanceo
from ypyquant.paper.almacen import AlmacenPaper
from ypyquant.paper.calendario import dias_pendientes, dias_pendientes_estimados, ultima_sesion_cerrada
from ypyquant.paper.servicio import ServicioPaper, TransicionInvalida

INICIO = "2021-01-04"


def _fuente(tickers=("AAA", "BBB"), fin="2021-03-31", deriva=0.002) -> FuenteManual:
    series = {t: barras("2019-01-01", fin, 50 + 25 * i, deriva) for i, t in enumerate([*tickers, "SPY"])}
    return FuenteManual(series)


def _receta(tickers=("AAA", "BBB"), estrategia="pesos_iguales", frecuencia="monthly", **extra) -> Receta:
    return Receta(estrategia=estrategia, parametros=extra.pop("parametros", {}), universo=list(tickers),
                  frecuencia_rebalanceo=frecuencia, comision=0.001, slippage=0.01, dias_historia_previa=60,
                  **extra).normalizada()


def _servicio(tmp_path, fuente, dia="2021-03-31", hora="20:00:00") -> ServicioPaper:
    return ServicioPaper(AlmacenPaper(tmp_path / "paper"), fuente, ahora=reloj(dia, hora))


# ---- 3.1 calendario -----------------------------------------------------------------------------


def test_ultima_sesion_cerrada_depende_de_la_hora_de_nueva_york():
    assert ultima_sesion_cerrada(reloj("2021-03-10", "12:00:00")()) == date(2021, 3, 9)
    assert ultima_sesion_cerrada(reloj("2021-03-10", "16:29:00")()) == date(2021, 3, 9)
    assert ultima_sesion_cerrada(reloj("2021-03-10", "16:31:00")()) == date(2021, 3, 10)


def test_dias_pendientes_solo_cerrados_y_posteriores_a_la_ultima_fecha():
    dias = pd.bdate_range("2021-03-01", "2021-03-12")
    pendientes = dias_pendientes(date(2021, 3, 4), dias, reloj("2021-03-10", "12:00:00")())
    assert pendientes[0] == pd.Timestamp("2021-03-05") and pendientes[-1] == pd.Timestamp("2021-03-09")
    assert dias_pendientes_estimados(date(2021, 3, 4), reloj("2021-03-10", "12:00:00")()) == 3  # 5, 8 y 9 de marzo
    assert dias_pendientes_estimados(date(2021, 3, 9), reloj("2021-03-10", "12:00:00")()) == 0


def test_rebalanceos_del_paper_coinciden_con_el_calendario_unico(tmp_path):
    servicio = _servicio(tmp_path, _fuente())
    sesion = servicio.crear_sesion(_receta(frecuencia="weekly"), fecha_alta=INICIO, modo_ejecucion="cierre")
    servicio.simulador.avanzar(sesion.id)
    tablas = servicio.almacen.leer_tablas(sesion.id)
    esperadas = fechas_de_rebalanceo(INICIO, "2021-03-31", "weekly", tablas.equity["fecha"])
    assert list(pd.to_datetime(tablas.pesos["fecha"]).unique()) == list(esperadas)


# ---- 3.4 apertura_siguiente ---------------------------------------------------------------------


def test_apertura_siguiente_decide_en_d_y_ejecuta_a_la_apertura_de_d_mas_1(tmp_path):
    fuente = _fuente()
    servicio = _servicio(tmp_path, fuente)
    sesion = servicio.crear_sesion(_receta(), fecha_alta=INICIO, capital_inicial=10_000.0, modo_ejecucion="apertura_siguiente")
    servicio.simulador.avanzar(sesion.id, hasta=date(2021, 1, 4))
    estado = servicio.almacen.leer_estado(sesion.id)
    tablas = servicio.almacen.leer_tablas(sesion.id)
    assert len(tablas.operaciones) == 0 and estado.pesos_pendientes is not None  # decidió, aún no ejecutó
    assert tablas.equity["equity"].iloc[0] == pytest.approx(10_000.0)  # la equity de D refleja la cartera anterior
    assert list(pd.to_datetime(tablas.pesos["fecha"]).unique()) == [pd.Timestamp("2021-01-04")]

    servicio.simulador.avanzar(sesion.id, hasta=date(2021, 1, 5))
    tablas = servicio.almacen.leer_tablas(sesion.id)
    assert set(pd.to_datetime(tablas.operaciones["fecha"])) == {pd.Timestamp("2021-01-05")}
    apertura = fuente.series["AAA"].loc["2021-01-05", "Open"]
    compra = tablas.operaciones[tablas.operaciones["ticker"] == "AAA"].iloc[0]
    assert compra["precio"] == pytest.approx(apertura * 1.01)  # slippage del 1 % sobre la apertura
    assert compra["comision"] == pytest.approx(compra["cantidad"] * compra["precio"] * 0.001)
    assert servicio.almacen.leer_estado(sesion.id).pesos_pendientes is None


def test_la_ultima_decision_queda_pendiente_al_acabar_los_datos(tmp_path):
    servicio = _servicio(tmp_path, _fuente(fin="2021-02-01"), dia="2021-02-01")  # el 1 de febrero es el primer día del mes
    sesion = servicio.crear_sesion(_receta(), fecha_alta=INICIO, modo_ejecucion="apertura_siguiente")
    servicio.simulador.avanzar(sesion.id)
    estado = servicio.almacen.leer_estado(sesion.id)
    assert estado.ultima_fecha_procesada == "2021-02-01"
    assert estado.pesos_pendientes is not None and estado.fecha_decision_pendiente == "2021-02-01"


# ---- 3.5 re-base por ajuste ---------------------------------------------------------------------


def test_rebase_por_dividendo_no_produce_salto_en_la_equity(tmp_path):
    fuente = _fuente(tickers=("AAA",))
    servicio = _servicio(tmp_path, fuente, dia="2021-01-20")
    sesion = servicio.crear_sesion(_receta(tickers=("AAA",)), fecha_alta=INICIO, modo_ejecucion="cierre")
    servicio.simulador.avanzar(sesion.id, hasta=date(2021, 1, 8))
    previa = servicio.almacen.leer_tablas(sesion.id).equity
    assert not [e for e in servicio.almacen.leer_tablas(sesion.id).eventos if e["tipo"] == "rebase"]  # sin cambios, nada

    df = fuente.series["AAA"]  # el histórico ajustado se reescala un 2 % (dividendo posterior al 8 de enero)
    df.loc[:"2021-01-08", ["Open", "High", "Low", "Close", "adjusted_close"]] *= 0.98
    servicio.simulador.avanzar(sesion.id, hasta=date(2021, 1, 11))
    tablas = servicio.almacen.leer_tablas(sesion.id)
    assert [e["tipo"] for e in tablas.eventos] == ["rebase"]
    rendimiento = tablas.equity["equity"].iloc[-1] / previa["equity"].iloc[-1]
    precio = df["Close"]
    assert rendimiento == pytest.approx(precio.loc["2021-01-11"] / precio.loc["2021-01-08"], rel=1e-9)  # sin salto del 2 %


# ---- 3.6 almacén, reanudación y bloqueo ---------------------------------------------------------


def test_caida_antes_del_commit_se_repite_el_dia_sin_duplicar_filas(tmp_path, monkeypatch):
    fuente = _fuente()
    limpio = _servicio(tmp_path / "limpio", fuente)
    s_limpia = limpio.crear_sesion(_receta(), nombre="a", fecha_alta=INICIO, modo_ejecucion="cierre")
    limpio.simulador.avanzar(s_limpia.id, hasta=date(2021, 1, 15))

    servicio = _servicio(tmp_path / "roto", fuente)
    sesion = servicio.crear_sesion(_receta(), nombre="a", fecha_alta=INICIO, modo_ejecucion="cierre")
    original = AlmacenPaper.guardar_estado
    llamadas = []

    def estado_que_falla(self, id_sesion, estado):
        llamadas.append(estado.ultima_fecha_procesada)
        if len(llamadas) == 4:  # las tablas del 4.º día ya están escritas, el commit no
            raise OSError("caída simulada")
        original(self, id_sesion, estado)

    monkeypatch.setattr(AlmacenPaper, "guardar_estado", estado_que_falla)
    resultado = servicio.simulador.avanzar(sesion.id, hasta=date(2021, 1, 15))
    assert resultado.error and "caída simulada" in resultado.error
    assert len(servicio.almacen.documentos.leer_parquet(sesion.id, "equity")) == 4  # resto del día no confirmado
    monkeypatch.undo()

    servicio.simulador.avanzar(sesion.id, hasta=date(2021, 1, 15))
    tablas, referencia = servicio.almacen.leer_tablas(sesion.id), limpio.almacen.leer_tablas(s_limpia.id)
    assert len(tablas.equity) == len(referencia.equity) == 10
    pd.testing.assert_frame_equal(tablas.equity, referencia.equity)
    pd.testing.assert_frame_equal(tablas.operaciones, referencia.operaciones)


def test_bloqueo_activo_omite_la_sesion_y_el_huerfano_se_recupera(tmp_path):
    servicio = _servicio(tmp_path, _fuente())
    sesion = servicio.crear_sesion(_receta(), fecha_alta=INICIO, modo_ejecucion="cierre")
    with servicio.almacen.bloqueo(sesion.id):  # alguien (otro hilo o proceso vivo) la está avanzando
        resultado = servicio.simulador.avanzar(sesion.id)
    assert resultado.omitida and resultado.omitida.startswith("bloqueada")
    assert servicio.almacen.leer_estado(sesion.id).ultima_fecha_procesada == "2021-01-03"

    muerto = subprocess.Popen([sys.executable, "-c", "pass"])
    muerto.wait()
    (servicio.almacen.documentos.carpeta(sesion.id) / "sesion.lock").write_text(str(muerto.pid))
    assert servicio.simulador.avanzar(sesion.id).dias_procesados > 0  # roba el bloqueo del proceso que ya no existe
    assert not (servicio.almacen.documentos.carpeta(sesion.id) / "sesion.lock").exists()


def test_pausar_reanudar_y_detener(tmp_path):
    servicio = _servicio(tmp_path, _fuente())
    sesion = servicio.crear_sesion(_receta(), fecha_alta=INICIO)
    servicio.cambiar_estado(sesion.id, "pausar")
    assert servicio.simulador.avanzar(sesion.id).omitida == "pausada"
    assert servicio.simulador.avanzar_todas() == []  # una pausada no entra en «todas»
    servicio.cambiar_estado(sesion.id, "reanudar")
    servicio.cambiar_estado(sesion.id, "detener")
    with pytest.raises(TransicionInvalida):
        servicio.cambiar_estado(sesion.id, "reanudar")


# ---- 3.7 día no cerrado, datos parciales, tickers que desaparecen --------------------------------


def test_no_procesa_el_dia_en_curso(tmp_path):
    servicio = _servicio(tmp_path, _fuente(), dia="2021-01-12", hora="12:00:00")  # el 12 aún no ha cerrado
    sesion = servicio.crear_sesion(_receta(), fecha_alta=INICIO, modo_ejecucion="cierre")
    servicio.simulador.avanzar(sesion.id)
    assert servicio.almacen.leer_estado(sesion.id).ultima_fecha_procesada == "2021-01-11"


def test_barra_parcial_del_ultimo_dia_se_pospone_y_se_completa_despues(tmp_path):
    fuente = _fuente(tickers=("AAA", "BBB", "CCC", "DDD", "EEE"))
    spy = fuente.series["SPY"]
    fuente.series["SPY"] = spy[spy.index < "2021-01-13"]  # el benchmark aún no trae la barra del 13
    servicio = _servicio(tmp_path, fuente, dia="2021-01-13")
    sesion = servicio.crear_sesion(_receta(tickers=("AAA", "BBB", "CCC", "DDD", "EEE")), fecha_alta=INICIO, modo_ejecucion="cierre")
    servicio.simulador.avanzar(sesion.id)
    assert servicio.almacen.leer_estado(sesion.id).ultima_fecha_procesada == "2021-01-12"
    fuente.series["SPY"] = spy  # llega la barra
    servicio.simulador.avanzar(sesion.id)
    assert servicio.almacen.leer_estado(sesion.id).ultima_fecha_procesada == "2021-01-13"


def test_ticker_que_desaparece_se_liquida_con_aviso(tmp_path):
    tickers = tuple(f"T{i}" for i in range(10))
    fuente = _fuente(tickers=tickers)
    corte = fuente.series["T9"]
    fuente.series["T9"] = corte[corte.index <= "2021-01-15"]  # T9 deja de cotizar el 15 de enero
    servicio = _servicio(tmp_path, fuente, dia="2021-02-05")
    sesion = servicio.crear_sesion(_receta(tickers=tickers, frecuencia="daily"), fecha_alta=INICIO, modo_ejecucion="cierre")
    servicio.simulador.avanzar(sesion.id)
    tablas = servicio.almacen.leer_tablas(sesion.id)
    liquidaciones = [e for e in tablas.eventos if e["tipo"] == "liquidacion"]
    assert len(liquidaciones) >= 1 and "T9" in liquidaciones[0]["mensaje"]
    assert pd.Timestamp(liquidaciones[0]["fecha"]) == pd.Timestamp("2021-01-22")  # 5 sesiones sin barra


# ---- 3.8 varias sesiones, una descarga -----------------------------------------------------------


def test_varias_sesiones_comparten_la_descarga_y_conservan_su_cartera(tmp_path):
    fuente = _fuente(tickers=("AAA", "BBB", "CCC"))
    servicio = _servicio(tmp_path, fuente)
    s1 = servicio.crear_sesion(_receta(tickers=("AAA", "BBB")), nombre="uno", fecha_alta=INICIO, modo_ejecucion="cierre")
    s2 = servicio.crear_sesion(_receta(tickers=("BBB", "CCC")), nombre="dos", fecha_alta=INICIO, modo_ejecucion="cierre")
    s3 = servicio.crear_sesion(_receta(tickers=("AAA", "CCC")), nombre="tres", fecha_alta=INICIO, modo_ejecucion="apertura_siguiente")
    servicio.simulador.avanzar_todas()
    assert fuente.peticiones == {"AAA": 1, "BBB": 1, "CCC": 1, "SPY": 1}
    carteras = [set(servicio.almacen.leer_estado(s.id).posiciones) for s in (s1, s2, s3)]
    assert carteras == [{"AAA", "BBB"}, {"BBB", "CCC"}, {"AAA", "CCC"}]


# ---- 3.9 estado de las estrategias registradas ---------------------------------------------------


@pytest.mark.lenta
@pytest.mark.parametrize("nombre", [e["nombre"] for e in listar_estrategias()])
def test_el_estado_de_cada_estrategia_registrada_es_json(tmp_path, nombre):
    try:
        obtener_estrategia(nombre).Parametros()
    except Exception:  # noqa: BLE001
        pytest.skip(f"{nombre} exige parámetros sin valor por defecto")
    receta = Receta(estrategia=nombre, universo=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"], frecuencia_rebalanceo="monthly",
                    dias_historia_previa=1100).normalizada()
    servicio = ServicioPaper(AlmacenPaper(tmp_path / "paper"), FuenteSintetica(), ahora=reloj("2021-03-12"))
    sesion = servicio.crear_sesion(receta, fecha_alta="2021-03-01", modo_ejecucion="cierre")
    resultado = servicio.simulador.avanzar(sesion.id)
    assert resultado.error is None and resultado.dias_procesados == 10
    reanudada = ServicioPaper(AlmacenPaper(tmp_path / "paper"), FuenteSintetica(), ahora=reloj("2021-03-19"))
    assert reanudada.simulador.avanzar(sesion.id).error is None  # recarga el estado desde el JSON y sigue


# ---- métricas forward = métricas del backtest ----------------------------------------------------


def test_las_metricas_forward_son_las_del_backtest_en_modo_cierre(tmp_path):
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    receta = Receta(estrategia="momentum", parametros={"top_k": 2, "meses_retorno": 2}, universo=tickers,
                    frecuencia_rebalanceo="monthly", dias_historia_previa=400).normalizada()
    servicio = ServicioPaper(AlmacenPaper(tmp_path / "paper"), FuenteSintetica(), ahora=reloj("2021-06-30"))
    sesion = servicio.crear_sesion(receta, fecha_alta=INICIO, modo_ejecucion="cierre")
    servicio.simulador.avanzar(sesion.id)
    config = ConfiguracionExperimento.desde_receta(receta, INICIO, "2021-06-30")
    backtest = ejecutar_con_datos(config, cargar_datos_experimento(config, FuenteSintetica()), completo=False).metricas
    forward = servicio.detalle(sesion.id)["metricas"]
    for clave in ("sharpe", "retorno_total", "max_drawdown", "volatilidad", "comisiones_totales", "n_operaciones"):
        assert forward[clave] == pytest.approx(backtest[clave], rel=1e-9), clave
    fila = servicio.listar()[0]
    assert fila["sharpe"] == pytest.approx(backtest["sharpe"]) and fila["dias"] == len(pd.bdate_range(INICIO, "2021-06-30"))
    assert np.isfinite(fila["costes"]) and fila["al_dia"] is True


def test_alta_rechaza_modo_invalido(tmp_path):  # el universo dinámico se admite: tests/test_paper_universo_vivo.py
    servicio = _servicio(tmp_path, _fuente())
    with pytest.raises(ValueError, match="Modo"):
        servicio.crear_sesion(_receta(), modo_ejecucion="intradia")


def test_alta_rechaza_apalancamiento_mayor_que_uno(tmp_path):
    servicio = _servicio(tmp_path, _fuente())
    receta = Receta.model_construct(**{**_receta().model_dump(), "parametros": {"apalancamiento_maximo": 1.5}})
    with pytest.raises(ValueError, match="margen"):
        servicio.crear_sesion(receta)
