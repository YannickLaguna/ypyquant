"""
test_paper_universo_vivo.py — El universo `sp500_pit` en el paper interno se resuelve en cada fecha procesada.

Responsabilidad, con membresías y precios sintéticos (sin red ni el JSON real):
    - el alta con `sp500_pit` se admite y conserva el nombre del universo (no una lista fija);
    - en cada fecha la estrategia solo ve a los miembros vigentes ese día: un valor que entra al índice pasa a la cartera
      en el siguiente rebalanceo, no antes;
    - un valor que sale del índice NO se vende el día de la salida: se mantiene hasta el siguiente rebalanceo, donde se
      liquida, y queda un aviso `salida_indice` el día de la salida;
    - avisos de calidad de las membresías: antigüedad (> 30 días respecto a la fecha procesada) y número de miembros
      lejos de ~500, una sola vez por sesión;
    - `detalle` expone `universo_dinamico` y `avisos_universo` vivos;
    - reanudar equivale a no parar (mismas operaciones avanzando en dos veces que de golpe).

Se relaciona con: ypyquant.paper.simulador/servicio/sesion, ypyquant.data.universos y estrategias/universo_dinamico.
"""

import pandas as pd

from tests._paper_utiles import FuenteManual, barras, reloj
from ypyquant.estrategias.receta import Receta
from ypyquant.paper.almacen import AlmacenPaper
from ypyquant.paper.servicio import ServicioPaper

ALTA = "2021-01-04"


def _m(alta, baja=None):
    return [{"index": "SP500", "date_added": alta, "date_removed": baja}]


# AAA y CCC siempre; BBB sale el 2021-02-15 (último día de miembro); DDD entra el 2021-02-01 (tras el rebalanceo de
# febrero del primer día de trading, 02-01, en el que ya es miembro: se incorpora ese día)
MEMBRESIAS = {"AAA": _m("2010-01-04"), "BBB": _m("2015-01-01", "2021-02-15"), "CCC": _m("2010-01-04"), "DDD": _m("2021-02-10")}


def _servicio(tmp_path, dia="2021-03-31") -> ServicioPaper:
    series = {t: barras("2019-01-01", "2021-04-30", 50 + 20 * i) for i, t in enumerate([*MEMBRESIAS, "SPY"])}
    return ServicioPaper(AlmacenPaper(tmp_path / "paper"), FuenteManual(series), ahora=reloj(dia), membresias=MEMBRESIAS,
                         meses_historia_pit=1)


def _receta() -> Receta:
    return Receta(estrategia="pesos_iguales", parametros={}, universo="sp500_pit", frecuencia_rebalanceo="monthly",
                  comision=0.001, slippage=0.001, dias_historia_previa=400).normalizada()


def _pesos_en(servicio, id_sesion, fecha) -> dict[str, float]:
    tabla = servicio.almacen.leer_tablas(id_sesion).pesos
    filas = tabla[pd.to_datetime(tabla["fecha"]) == pd.Timestamp(fecha)]
    return dict(zip(filas["ticker"], filas["peso"], strict=True))


def test_el_alta_con_sp500_pit_conserva_el_nombre_del_universo(tmp_path):
    servicio = _servicio(tmp_path)
    sesion = servicio.crear_sesion(_receta(), fecha_alta=ALTA, modo_ejecucion="cierre")
    assert sesion.receta.universo == "sp500_pit" and sesion.universo_dinamico


def test_la_estrategia_solo_ve_a_los_miembros_vigentes_de_cada_fecha(tmp_path):
    servicio = _servicio(tmp_path)
    sesion = servicio.crear_sesion(_receta(), fecha_alta=ALTA, modo_ejecucion="cierre")
    servicio.simulador.avanzar(sesion.id)
    assert set(_pesos_en(servicio, sesion.id, "2021-01-04")) == {"AAA", "BBB", "CCC"}  # DDD aún no es miembro
    assert set(_pesos_en(servicio, sesion.id, "2021-02-01")) == {"AAA", "BBB", "CCC"}  # entra el 02-10: no antes
    assert set(_pesos_en(servicio, sesion.id, "2021-03-01")) == {"AAA", "CCC", "DDD"}  # BBB ya salió, DDD ya entró


def test_el_valor_que_sale_se_liquida_en_el_siguiente_rebalanceo_con_aviso(tmp_path):
    servicio = _servicio(tmp_path)
    sesion = servicio.crear_sesion(_receta(), fecha_alta=ALTA, modo_ejecucion="cierre")
    servicio.simulador.avanzar(sesion.id)
    tablas = servicio.almacen.leer_tablas(sesion.id)
    ventas_bbb = tablas.operaciones[(tablas.operaciones["ticker"] == "BBB") & (tablas.operaciones["lado"] == "venta")]
    assert [f"{f:%Y-%m-%d}" for f in ventas_bbb["fecha"]] == ["2021-03-01"]  # no el 02-16, día en que sale del índice
    salidas = [e for e in tablas.eventos if e["tipo"] == "salida_indice"]
    assert len(salidas) == 1 and salidas[0]["fecha"] == "2021-02-16" and "BBB" in salidas[0]["mensaje"]
    assert "BBB" not in servicio.almacen.leer_estado(sesion.id).posiciones


def test_avisos_de_calidad_de_las_membresias_una_sola_vez(tmp_path):
    servicio = _servicio(tmp_path)
    sesion = servicio.crear_sesion(_receta(), fecha_alta=ALTA, modo_ejecucion="cierre")
    servicio.simulador.avanzar(sesion.id)
    tipos = [e["tipo"] for e in servicio.almacen.leer_tablas(sesion.id).eventos]
    assert tipos.count("membresia_numero") == 1  # 3-4 miembros frente a ~500
    assert tipos.count("membresia_antigua") == 1  # la última membresía es de febrero y se procesa hasta el 31 de marzo
    avisos = servicio.detalle(sesion.id)["avisos_universo"]
    assert any("antigüedad" in a for a in avisos) and servicio.detalle(sesion.id)["universo_dinamico"] is True


def test_reanudar_equivale_a_no_parar_con_universo_dinamico(tmp_path):
    de_golpe = _servicio(tmp_path / "a")
    s1 = de_golpe.crear_sesion(_receta(), fecha_alta=ALTA, modo_ejecucion="apertura_siguiente")
    de_golpe.simulador.avanzar(s1.id)
    partido = _servicio(tmp_path / "b", dia="2021-02-19")
    s2 = partido.crear_sesion(_receta(), fecha_alta=ALTA, modo_ejecucion="apertura_siguiente")
    partido.simulador.avanzar(s2.id)
    partido.simulador.ahora = reloj("2021-03-31")
    partido.simulador.avanzar(s2.id)
    columnas = ["fecha", "ticker", "lado", "cantidad", "precio"]
    pd.testing.assert_frame_equal(de_golpe.almacen.leer_tablas(s1.id).operaciones[columnas].reset_index(drop=True),
                                  partido.almacen.leer_tablas(s2.id).operaciones[columnas].reset_index(drop=True))
