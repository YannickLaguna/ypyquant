"""
Tests del índice por receta del almacén de experimentos (`AlmacenExperimentos.resumenes_de_receta`).

Responsabilidad: comprueba que «los backtests de esta receta» se responde sin recorrer el disco entero, que el índice
se mantiene al registrar y al borrar, y que ignora configuraciones ajenas o dañadas (importadas del TFM).
Se relaciona con: ypyquant.estrategias.almacen (índice), .receta (huella de receta) y
tests/test_servicio_experimentos.py.
"""

from __future__ import annotations

import json

from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.experimentos import ConfiguracionExperimento


def _config(**cambios) -> ConfiguracionExperimento:
    base = {"estrategia": "momentum", "parametros": {"top_k": 3}, "universo": ["AAA", "BBB", "CCC"],
            "fecha_inicio": "2020-01-01", "fecha_fin": "2021-01-01"}
    return ConfiguracionExperimento(**{**base, **cambios}).normalizada()


def test_agrupa_por_receta_sin_importar_periodo_ni_capital(tmp_path):
    almacen = AlmacenExperimentos(tmp_path)
    a = almacen.registrar(_config())
    b = almacen.registrar(_config(fecha_inicio="2018-01-01", capital_inicial=5_000.0))
    otra = almacen.registrar(_config(parametros={"top_k": 2}))
    huella = _config().huella_receta()
    assert {r["id"] for r in almacen.resumenes_de_receta(huella)} == {a, b}
    assert otra not in {r["id"] for r in almacen.resumenes_de_receta(huella)}


def test_se_mantiene_al_registrar_y_borrar_despues_de_construirse(tmp_path):
    almacen = AlmacenExperimentos(tmp_path)
    huella = _config().huella_receta()
    assert almacen.resumenes_de_receta(huella) == []  # índice ya construido (vacío)
    id_nuevo = almacen.registrar(_config())
    assert [r["id"] for r in almacen.resumenes_de_receta(huella)] == [id_nuevo]
    almacen.borrar(id_nuevo)
    assert almacen.resumenes_de_receta(huella) == []


def test_se_construye_desde_el_disco_e_ignora_configuraciones_ajenas_o_danadas(tmp_path):
    previo = AlmacenExperimentos(tmp_path)
    id_ok = previo.registrar(_config())
    for nombre, contenido in (("tfm-ajeno", {"estrategia": "tfm_x", "universo": "tfm:N=1"}), ("roto", "no es json")):
        (tmp_path / nombre).mkdir()
        (tmp_path / nombre / "config.json").write_text(json.dumps(contenido), encoding="utf-8")
    nuevo = AlmacenExperimentos(tmp_path)  # arranque: construye el índice leyendo el disco
    assert [r["id"] for r in nuevo.resumenes_de_receta(_config().huella_receta())] == [id_ok]


def test_la_consulta_no_recorre_el_almacen(tmp_path, monkeypatch):
    almacen = AlmacenExperimentos(tmp_path)
    for k in range(1, 6):
        almacen.registrar(_config(parametros={"top_k": k}))
    huella = _config(parametros={"top_k": 2}).huella_receta()
    almacen.indexar()
    llamadas = []
    original = almacen._resumen
    monkeypatch.setattr(almacen, "_resumen", lambda carpeta: llamadas.append(carpeta.name) or original(carpeta))
    assert len(almacen.resumenes_de_receta(huella)) == 1
    assert len(llamadas) == 1  # solo se lee el experimento de la receta, no los otros cuatro
