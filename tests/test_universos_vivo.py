"""
test_universos_vivo.py — Membresía VIGENTE del S&P 500 en una fecha y avisos de calidad de las membresías.

Responsabilidad: con membresías sintéticas comprueba `miembros_vigentes_en` (intervalos inclusivos, como la máscara
del backtest), `ultima_fecha_membresias`, y `avisos_membresias`: aviso de antigüedad (> 30 días respecto a la fecha
procesada, con instrucciones para refrescar) y aviso de número de miembros lejos de ~500.

Se relaciona con: ypyquant.data.universos y ypyquant.paper.simulador (que usa estos avisos en el universo en vivo).
"""

import pandas as pd

from ypyquant.data.universos import (
    MIEMBROS_ESPERADOS,
    avisos_membresias,
    miembros_vigentes_en,
    ultima_fecha_membresias,
)


def _m(alta, baja=None):
    return [{"index": "SP500", "date_added": alta, "date_removed": baja}]


MEMBRESIAS = {"AAA": _m("2010-01-04"), "BBB": _m("2015-01-01", "2020-06-30"), "CCC": _m("2020-07-15"), "DDD": _m(None, "2001-01-01")}


def test_vigentes_son_los_miembros_de_esa_fecha():
    assert miembros_vigentes_en(MEMBRESIAS, "2020-06-30") == ["AAA", "BBB"]  # el último día de BBB aún es miembro
    assert miembros_vigentes_en(MEMBRESIAS, "2020-07-01") == ["AAA"]
    assert miembros_vigentes_en(MEMBRESIAS, "2020-07-15") == ["AAA", "CCC"]  # el día de alta ya cuenta
    assert miembros_vigentes_en(MEMBRESIAS, pd.Timestamp("2000-01-01")) == ["DDD"]


def test_ultima_fecha_de_las_membresias_mira_altas_y_bajas():
    assert ultima_fecha_membresias(MEMBRESIAS) == pd.Timestamp("2020-07-15")
    assert ultima_fecha_membresias({}) is None


def test_aviso_de_antiguedad_a_partir_de_30_dias():
    assert not [a for a in avisos_membresias(MEMBRESIAS, "2020-08-14", esperados=None) if "antigüedad" in a]  # 30 días justos
    avisos = avisos_membresias(MEMBRESIAS, "2020-08-15", esperados=None)  # 31 días
    assert len(avisos) == 1 and "31 días" in avisos[0] and "exportar_membresias_sp500.py" in avisos[0]


def test_una_fecha_anterior_a_la_ultima_membresia_no_avisa_de_antiguedad():
    assert avisos_membresias(MEMBRESIAS, "2019-01-01", esperados=None) == []


def test_aviso_de_numero_de_miembros_lejos_de_lo_esperado():
    muchas = {f"T{i}": _m("2000-01-03") for i in range(MIEMBROS_ESPERADOS + 25)}
    avisos = avisos_membresias(muchas, "2000-02-01")
    assert len(avisos) == 1 and "525" in avisos[0] and "bajas" in avisos[0]
    normales = {f"T{i}": _m("2000-01-03") for i in range(MIEMBROS_ESPERADOS + 3)}
    assert avisos_membresias(normales, "2000-02-01") == []
