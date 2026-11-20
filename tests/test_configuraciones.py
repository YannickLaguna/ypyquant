"""
test_configuraciones.py — Tests del modelo y el almacén de configuraciones guardadas (fase 1 del plan).

Se relaciona con: src/ypyquant/estrategias/configuraciones.py, .experimentos (ConfiguracionExperimento y su huella)
y tests/test_api_configuraciones.py (las rutas).
"""

import pytest
from pydantic import ValidationError

from ypyquant.estrategias.configuraciones import (
    AlmacenConfiguraciones,
    ConfiguracionExistente,
    ConfiguracionGuardada,
    OrigenConfiguracion,
    crear_slug,
)
from ypyquant.estrategias.experimentos import ConfiguracionExperimento

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _experimento(**cambios) -> ConfiguracionExperimento:
    base = {"estrategia": "momentum", "parametros": {"top_k": 3}, "universo": TICKERS,
            "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}
    return ConfiguracionExperimento(**{**base, **cambios}).normalizada()


def _guardada(**cambios) -> ConfiguracionGuardada:
    return ConfiguracionGuardada.desde_experimento(_experimento(**cambios), nombre="Momentum top 3")


# ---- modelo ----------------------------------------------------------------------------------


def test_huella_de_la_receta_ignora_fechas_y_capital_pero_no_los_parametros():
    a = _guardada()
    b = _guardada(fecha_inicio="2015-01-01", fecha_fin="2019-12-31", capital_inicial=5_000.0)
    c = _guardada(parametros={"top_k": 4})
    assert a.huella_estrategia == b.huella_estrategia
    assert a.huella_estrategia != c.huella_estrategia
    assert len(a.huella_estrategia) == 64


def test_a_experimento_reproduce_la_huella_de_la_configuracion_original():
    original = _experimento(capital_inicial=50_000.0, comision=0.0005)
    guardada = ConfiguracionGuardada.desde_experimento(original, nombre="x")
    reconstruida = guardada.a_experimento(original.fecha_inicio, original.fecha_fin)
    assert reconstruida.huella() == original.huella()
    # Con otras fechas es otro experimento, pero la receta es la misma
    otra = guardada.a_experimento("2018-01-01", "2019-01-01")
    assert otra.huella() != original.huella() and otra.huella_receta() == original.huella_receta()


def test_a_experimento_admite_otro_capital():
    guardada = _guardada()
    assert guardada.a_experimento("2020-01-01", "2021-01-01", capital_inicial=1_000.0).capital_inicial == 1_000.0


def test_universo_con_nombre_se_congela_como_lista_y_conserva_la_etiqueta():
    guardada = ConfiguracionGuardada.desde_experimento(_experimento(universo="sp500:3"), nombre="Con nombre")
    assert isinstance(guardada.universo, list) and len(guardada.universo) == 3
    assert guardada.universo_etiqueta == "sp500:3"
    assert guardada.a_experimento("2020-01-01", "2021-01-01").universo == guardada.universo


def test_universo_point_in_time_se_conserva_como_nombre_porque_es_dinamico():
    guardada = ConfiguracionGuardada.desde_experimento(_experimento(universo="sp500_pit"), nombre="PIT")
    assert guardada.universo == "sp500_pit" and guardada.universo_etiqueta == "sp500_pit"


def test_parametros_invalidos_se_rechazan_al_crear():
    with pytest.raises(ValidationError):
        ConfiguracionGuardada.desde_experimento(
            ConfiguracionExperimento(estrategia="momentum", parametros={"top_k": 0}, universo=TICKERS,
                                     fecha_inicio="2020-01-01", fecha_fin="2021-12-31"), nombre="mala")


def test_origen_conserva_metricas_del_momento_de_guardar():
    origen = OrigenConfiguracion(tipo="experimento", id_experimento="momentum-abc", metricas={"sharpe": 1.2},
                                 fechas=["2020-01-01", "2021-12-31"])
    guardada = ConfiguracionGuardada.desde_experimento(_experimento(), nombre="Con origen", origen=origen)
    assert guardada.origen.metricas["sharpe"] == 1.2 and guardada.origen.tipo == "experimento"


def test_crear_slug_estable_y_sin_caracteres_raros():
    assert crear_slug("Momentum top-5 (ejemplo)") == "momentum-top-5-ejemplo"
    assert crear_slug("  Ñandú  ") == "nandu"
    with pytest.raises(ValueError):
        crear_slug("¿¿??")


# ---- almacén ---------------------------------------------------------------------------------


def test_almacen_guardar_leer_listar(tmp_path):
    almacen = AlmacenConfiguraciones(tmp_path)
    guardada = almacen.guardar(_guardada())
    assert guardada.id == "momentum-top-3" and guardada.version == 1
    assert almacen.leer("momentum-top-3").huella_estrategia == guardada.huella_estrategia
    assert [c.id for c in almacen.listar()] == ["momentum-top-3"]
    assert almacen.listar(estrategia="otra") == []
    assert [c.id for c in almacen.listar(texto="TOP")] == ["momentum-top-3"]


def test_almacen_id_duplicado_es_error(tmp_path):
    almacen = AlmacenConfiguraciones(tmp_path)
    almacen.guardar(_guardada())
    with pytest.raises(ConfiguracionExistente):
        almacen.guardar(_guardada())


def test_nueva_version_no_modifica_la_anterior(tmp_path):
    almacen = AlmacenConfiguraciones(tmp_path)
    v1 = almacen.guardar(_guardada())
    v2 = almacen.nueva_version(v1.id, {"parametros": {"top_k": 4}, "notas": "más diversificada"})
    assert v2.version == 2 and v2.parametros["top_k"] == 4 and v2.deriva_de == 1
    assert almacen.leer(v1.id, version=1).parametros["top_k"] == 3  # inmutable
    assert almacen.leer(v1.id).version == 2  # sin versión = la última
    assert v2.huella_estrategia != v1.huella_estrategia
    assert [v.version for v in almacen.versiones(v1.id)] == [1, 2]
    assert [c.version for c in almacen.listar()] == [2]  # el listado muestra solo la última de cada una


def test_nueva_version_valida_parametros_y_campos(tmp_path):
    almacen = AlmacenConfiguraciones(tmp_path)
    almacen.guardar(_guardada())
    with pytest.raises(ValidationError):
        almacen.nueva_version("momentum-top-3", {"parametros": {"top_k": 0}})
    with pytest.raises(ValueError, match="campo"):
        almacen.nueva_version("momentum-top-3", {"estrategia": "pesos_iguales"})  # la estrategia no se cambia


def test_archivar_oculta_del_listado_sin_borrar(tmp_path):
    almacen = AlmacenConfiguraciones(tmp_path)
    almacen.guardar(_guardada())
    almacen.archivar("momentum-top-3")
    assert almacen.listar() == []
    assert [c.id for c in almacen.listar(archivadas=True)] == ["momentum-top-3"]
    assert almacen.leer("momentum-top-3").archivada is True
    almacen.archivar("momentum-top-3", False)
    assert [c.id for c in almacen.listar()] == ["momentum-top-3"]


def test_leer_inexistente_y_ids_peligrosos(tmp_path):
    almacen = AlmacenConfiguraciones(tmp_path)
    with pytest.raises(KeyError):
        almacen.leer("no-existe")
    with pytest.raises(ValueError):
        almacen.leer("../fuera")
    almacen.guardar(_guardada())
    with pytest.raises(KeyError):
        almacen.leer("momentum-top-3", version=9)


def test_borrar(tmp_path):
    almacen = AlmacenConfiguraciones(tmp_path)
    almacen.guardar(_guardada())
    assert almacen.borrar("momentum-top-3") is True
    assert almacen.borrar("momentum-top-3") is False
    assert almacen.listar() == []


def test_referencia_a_un_marco_es_opcional_y_se_conserva(tmp_path):
    almacen = AlmacenConfiguraciones(tmp_path)
    con_marco = ConfiguracionGuardada.desde_experimento(_experimento(), nombre="Con marco", id_marco="marco-1")
    almacen.guardar(con_marco)
    assert almacen.leer("con-marco").id_marco == "marco-1"
    almacen.guardar(_guardada())
    assert almacen.leer("momentum-top-3").id_marco is None
