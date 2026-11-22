"""
Pruebas del módulo común `almacen_documentos` y de `raiz_datos`.

Responsabilidad: fija la interfaz del almacén de documentos por id (carpeta por documento, JSON y Parquet atómicos,
validación de ids, marca, listado, borrado, archivos planos) sin ningún dominio por encima, y su resistencia a los
bloqueos transitorios de Windows y a la escritura concurrente. También comprueba cómo `RaizDatos` construye las rutas.

Se relaciona con: ypyquant.almacen_documentos, ypyquant.raiz_datos y tests/test_formato_en_disco.py (el formato histórico).
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pandas as pd
import pytest

from ypyquant import almacen_documentos
from ypyquant.almacen_documentos import (
    AlmacenDocumentos,
    ColeccionJson,
    escribir_json_atomico,
    id_sin_separadores,
    id_slug,
    leer_json,
    limpiar_para_json,
)
from ypyquant.raiz_datos import RAIZ_REPO, RaizDatos


def test_ciclo_de_un_documento_json_y_parquet(tmp_path):
    almacen = AlmacenDocumentos(tmp_path / "docs", nombre="sesión")
    assert almacen.ids() == [] and not almacen.existe("a")
    carpeta = almacen.crear("a")
    assert carpeta == tmp_path / "docs" / "a" and almacen.existe("a") and almacen.hay_carpeta("a")
    almacen.escribir_json("a", "datos", {"ñ": np.float64("nan"), "n": np.int64(3), "x": [np.float64(1.5)]})
    assert almacen.leer_json("a", "datos") == {"ñ": None, "n": 3, "x": [1.5]}
    assert "ñ" in (carpeta / "datos.json").read_text(encoding="utf-8")  # sin escapar tildes
    almacen.escribir_parquet("a", "tabla", pd.DataFrame({"x": [1, 2]}))
    assert list(almacen.leer_parquet("a", "tabla")["x"]) == [1, 2] and almacen.tiene_parquet("a", "tabla")
    assert almacen.leer_parquet("a", "otra") is None and almacen.leer_json("a", "otra") is None
    assert almacen.nombres_json("a") == ["datos"] and almacen.tiene_json("a", "datos")
    assert not list(carpeta.glob("*.tmp"))
    assert almacen.ids() == ["a"] and almacen.borrar("a") is True and almacen.borrar("a") is False


def test_crear_con_reemplazar_vacia_la_carpeta(tmp_path):
    almacen = AlmacenDocumentos(tmp_path)
    almacen.crear("a")
    almacen.escribir_json("a", "viejo", {})
    almacen.crear("a")  # sin reemplazar conserva
    assert almacen.tiene_json("a", "viejo")
    almacen.crear("a", reemplazar=True)
    assert not almacen.tiene_json("a", "viejo")


@pytest.mark.parametrize("malo", ["", "../x", "a/b", "a\\b", ".oculto"])
def test_ids_invalidos_se_rechazan_en_toda_operacion(tmp_path, malo):
    almacen = AlmacenDocumentos(tmp_path, nombre="experimento")
    for operacion in (almacen.carpeta, almacen.existe, almacen.borrar, almacen.crear,
                      lambda i: almacen.leer_json(i, "x")):
        with pytest.raises(ValueError, match="Identificador de experimento inválido"):
            operacion(malo)


def test_validadores_de_id(tmp_path):
    assert id_sin_separadores("momentum-2edd1300a9") and not id_sin_separadores("a/b")
    assert id_slug("mi-config_1.v2") and not id_slug("Mayus") and not id_slug("a..b") and not id_slug("-x")
    almacen = AlmacenDocumentos(tmp_path, nombre="marco", validar_id=id_slug)
    with pytest.raises(ValueError, match="marco"):
        almacen.carpeta("Mayúscula")
    assert almacen.carpeta("ok-1") == tmp_path / "ok-1"


def test_marca_define_que_carpeta_es_un_documento(tmp_path):
    almacen = AlmacenDocumentos(tmp_path, marca="principal")
    almacen.crear("a")
    almacen.crear("b")
    almacen.escribir_json("a", "principal", {})
    assert almacen.ids() == ["a"] and almacen.existe("a") and not almacen.existe("b") and almacen.hay_carpeta("b")
    (tmp_path / "suelto.txt").write_text("no es un documento")
    assert almacen.ids() == ["a"]


def test_escritura_atomica_reintenta_ante_permission_error(tmp_path, monkeypatch):
    real = os.replace
    fallos = {"n": 0}

    def inestable(origen, destino):
        if fallos["n"] < 3:
            fallos["n"] += 1
            raise PermissionError("bloqueado por el antivirus")
        return real(origen, destino)

    monkeypatch.setattr(almacen_documentos.os, "replace", inestable)
    monkeypatch.setattr(almacen_documentos.time, "sleep", lambda _s: None)
    escribir_json_atomico(tmp_path / "a.json", {"x": 1})
    assert fallos["n"] == 3 and leer_json(tmp_path / "a.json") == {"x": 1}


def test_escritura_atomica_se_rinde_tras_los_reintentos(tmp_path, monkeypatch):
    def siempre_bloqueado(*_args):
        raise PermissionError("siempre")

    monkeypatch.setattr(almacen_documentos.os, "replace", siempre_bloqueado)
    monkeypatch.setattr(almacen_documentos.time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        escribir_json_atomico(tmp_path / "a.json", {})


def test_leer_json_con_reintento_espera_a_que_aparezca(tmp_path, monkeypatch):
    ruta = tmp_path / "estado.json"
    esperas = []

    def dormir(segundos):
        esperas.append(segundos)
        if len(esperas) == 2:
            ruta.write_text(json.dumps({"ok": True}), encoding="utf-8")

    monkeypatch.setattr(almacen_documentos.time, "sleep", dormir)
    assert leer_json(ruta) is None and esperas == []  # sin reintento no espera
    assert leer_json(ruta, reintentar_si_falta=True) == {"ok": True} and len(esperas) == 2


def test_escrituras_concurrentes_del_mismo_archivo_no_corrompen(tmp_path):
    almacen = AlmacenDocumentos(tmp_path)
    almacen.crear("a")
    errores = []

    def escribir(valor):
        try:
            for i in range(30):
                almacen.escribir_json("a", "c", {"valor": valor, "i": i, "relleno": "x" * 2000})
                assert almacen.leer_json("a", "c")["relleno"] == "x" * 2000
        except Exception as error:  # noqa: BLE001
            errores.append(error)

    hilos = [threading.Thread(target=escribir, args=(v,)) for v in range(4)]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    assert errores == [] and not list((tmp_path / "a").glob("*.tmp"))


def test_limpiar_para_json():
    assert limpiar_para_json({"a": float("inf"), "b": (np.bool_(True), np.int32(2))}) == {"a": None, "b": [True, 2]}


def test_coleccion_json_de_archivos_planos(tmp_path):
    coleccion = ColeccionJson(tmp_path / "cmp", prefijo="cmp-")
    assert coleccion.ids() == [] and coleccion.leer("cmp-a") is None
    coleccion.escribir("cmp-b", {"x": 2})
    coleccion.escribir("cmp-a", {"x": 1})
    (tmp_path / "cmp" / "otro.json").write_text("{}", encoding="utf-8")
    assert coleccion.ids() == ["cmp-a", "cmp-b"] and coleccion.listar() == [{"x": 1}, {"x": 2}]


def test_raiz_datos_construye_las_rutas_hermanas(tmp_path):
    raiz = RaizDatos(tmp_path)
    assert raiz.experimentos == tmp_path / "experimentos" and raiz.barridos == tmp_path / "barridos"
    assert raiz.configuraciones == tmp_path / "configuraciones" and raiz.marcos == tmp_path / "marcos"
    assert raiz.reservado == tmp_path / "reservado.json"
    desde = RaizDatos.de_experimentos(tmp_path / "exp")
    assert desde.experimentos == tmp_path / "exp" and desde.marcos == tmp_path / "marcos"
    assert RaizDatos().base == RAIZ_REPO / "data"
