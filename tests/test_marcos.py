"""
test_marcos.py — Tests del «marco» de comparación ceteris paribus: modelo, `aplicar_marco`, almacén de marcos y
comparativas, y la función pura `diferencias_ceteris_paribus`.

Se relaciona con: src/ypyquant/estrategias/marcos.py, .experimentos (huella) y tests/test_api_marcos.py (rutas).
"""

import pytest
from pydantic import ValidationError

from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.estrategias.marcos import (
    AlmacenMarcos,
    MarcoComparacion,
    MarcoExistente,
    aplicar_marco,
    diferencias_ceteris_paribus,
    preparar_comparativa,
)

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _marco(**cambios) -> MarcoComparacion:
    base = {"id": "marco-prueba", "nombre": "Marco de prueba", "universo": TICKERS, "fecha_inicio": "2020-01-01",
            "fecha_fin": "2021-12-31", "frecuencia_rebalanceo": "monthly", "comision": 0.0005, "slippage": 0.0002,
            "benchmark": "SPY", "capital_inicial": 50_000.0, "semilla": 7, "dias_historia_previa": 900,
            "parametros_comunes": {"ventana_dias": 504, "peso_maximo": 0.5}}
    return MarcoComparacion(**{**base, **cambios})


# ---- modelo ----------------------------------------------------------------------------------


def test_marco_valida_fechas_y_normaliza_el_universo():
    marco = _marco(universo=["bbb", "aaa", "bbb"])
    assert marco.universo == ["BBB", "AAA"]
    with pytest.raises(ValidationError):
        _marco(fecha_inicio="2022-01-01")
    with pytest.raises(ValidationError):
        _marco(universo=[])


def test_universo_con_nombre_se_congela_y_pit_se_conserva():
    congelado = _marco(universo="sp500:4")
    assert isinstance(congelado.universo, list) and len(congelado.universo) == 4 and congelado.universo_etiqueta == "sp500:4"
    pit = _marco(universo="sp500_pit")
    assert pit.universo == "sp500_pit" and pit.universo_etiqueta == "sp500_pit"


# ---- aplicar_marco ---------------------------------------------------------------------------


def test_aplicar_marco_impone_el_entorno_y_los_comunes_que_la_estrategia_declara():
    config, informe = aplicar_marco(_marco(), "bayes_stein", {})
    assert isinstance(config, ConfiguracionExperimento)
    assert (config.fecha_inicio, config.fecha_fin, config.frecuencia_rebalanceo) == ("2020-01-01", "2021-12-31", "monthly")
    assert (config.comision, config.slippage, config.capital_inicial, config.semilla) == (0.0005, 0.0002, 50_000.0, 7)
    assert config.dias_historia_previa == 900 and sorted(config.universo) == TICKERS
    assert config.parametros["ventana_dias"] == 504 and config.parametros["peso_maximo"] == 0.5
    assert informe.aplicados == {"ventana_dias": 504, "peso_maximo": 0.5} and informe.no_aplicados == []


def test_aplicar_marco_informa_de_los_comunes_que_la_estrategia_no_tiene():
    config, informe = aplicar_marco(_marco(), "momentum", {"top_k": 3})
    assert config.parametros["top_k"] == 3 and "ventana_dias" not in config.parametros
    assert informe.aplicados == {} and sorted(informe.no_aplicados) == ["peso_maximo", "ventana_dias"]
    assert any("no se aplic" in aviso.lower() for aviso in informe.avisos)


def test_el_marco_prevalece_sobre_un_parametro_propio_que_es_comun():
    config, informe = aplicar_marco(_marco(), "bayes_stein", {"ventana_dias": 252, "penalizacion_turnover": 1.0})
    assert config.parametros["ventana_dias"] == 504 and config.parametros["penalizacion_turnover"] == 1.0
    assert informe.sobrescritos == ["ventana_dias"]


def test_aplicar_marco_con_parametros_propios_invalidos_falla():
    with pytest.raises(ValidationError):
        aplicar_marco(_marco(), "momentum", {"top_k": 0})


def test_aplicar_marco_estrategia_inexistente_o_tfm_importada():
    with pytest.raises(ValueError, match="no se puede lanzar"):
        aplicar_marco(_marco(), "tfm_bayes_stein", {})
    with pytest.raises(ValueError, match="no se puede lanzar"):
        aplicar_marco(_marco(), "no_existe", {})


def test_mismo_marco_y_estrategia_dan_la_misma_huella_y_comunes_distintos_otra():
    a, _ = aplicar_marco(_marco(), "pesos_iguales", {})
    b, _ = aplicar_marco(_marco(notas="otra nota", nombre="Otro nombre"), "pesos_iguales", {})
    c, _ = aplicar_marco(_marco(comision=0.002), "pesos_iguales", {})
    assert a.huella() == b.huella() and a.huella() != c.huella()


def test_marco_sin_frecuencia_deja_cada_estrategia_con_la_suya_y_avisa():
    _, informe = aplicar_marco(_marco(frecuencia_rebalanceo=None), "momentum", {})
    assert any("frecuencia" in aviso for aviso in informe.avisos)


def test_estimadores_comunes_nuevos_se_aplican_a_los_estimadores_robustos():
    marco = _marco(parametros_comunes={"estimador_sigma": "muestral", "frecuencia_estimacion": "mensual"})
    config, _informe = aplicar_marco(marco, "bayes_stein", {})
    assert config.parametros["frecuencia_estimacion"] == "mensual" and config.parametros["estimador_sigma"] == "muestral"
    _, informe_pi = aplicar_marco(marco, "pesos_iguales", {})
    assert sorted(informe_pi.no_aplicados) == ["estimador_sigma", "frecuencia_estimacion"]


def test_un_comun_con_valor_invalido_para_una_estrategia_falla_con_mensaje():
    with pytest.raises(ValidationError):
        aplicar_marco(_marco(parametros_comunes={"peso_maximo": 5.0}), "bayes_stein", {})


# ---- preparar_comparativa --------------------------------------------------------------------


def test_preparar_comparativa_valida_todo_antes_de_lanzar():
    entradas = [{"estrategia": "pesos_iguales"}, {"estrategia": "momentum", "parametros": {"top_k": 3}}]
    preparadas = preparar_comparativa(_marco(), entradas)
    assert [c.estrategia for c, _ in preparadas] == ["pesos_iguales", "momentum"]
    with pytest.raises(ValueError, match="momentum"):
        preparar_comparativa(_marco(), [{"estrategia": "pesos_iguales"}, {"estrategia": "momentum", "parametros": {"top_k": 0}}])
    with pytest.raises(ValueError, match="al menos"):
        preparar_comparativa(_marco(), [])


# ---- almacén ---------------------------------------------------------------------------------


def test_almacen_crear_leer_listar_actualizar_borrar(tmp_path):
    almacen = AlmacenMarcos(tmp_path)
    almacen.guardar(_marco())
    assert almacen.leer("marco-prueba").comision == 0.0005
    assert [m.id for m in almacen.listar()] == ["marco-prueba"]
    with pytest.raises(MarcoExistente):
        almacen.guardar(_marco())
    almacen.actualizar(_marco(comision=0.003))
    assert almacen.leer("marco-prueba").comision == 0.003
    with pytest.raises(KeyError):
        almacen.actualizar(_marco(id="otro"))
    with pytest.raises(ValueError):
        almacen.leer("../x")
    assert almacen.borrar("marco-prueba") is True and almacen.borrar("marco-prueba") is False


def test_comparativas_se_registran_de_forma_idempotente(tmp_path):
    almacen = AlmacenMarcos(tmp_path)
    marco = _marco()
    almacen.guardar(marco)
    entradas = [{"id_experimento": "a-1", "estrategia": "pesos_iguales", "parametros": {}, "informe": {}, "reutilizado": False},
                {"id_experimento": "b-2", "estrategia": "momentum", "parametros": {"top_k": 3}, "informe": {}, "reutilizado": True}]
    primera = almacen.registrar_comparativa(marco, entradas)
    segunda = almacen.registrar_comparativa(marco, list(reversed(entradas)))
    assert primera["id"] == segunda["id"] and primera["id"].startswith("cmp-")
    assert almacen.leer_comparativa(primera["id"])["marco"]["id"] == "marco-prueba"
    assert [c["id"] for c in almacen.listar_comparativas("marco-prueba")] == [primera["id"]]
    assert almacen.listar_comparativas("otro") == []
    with pytest.raises(KeyError):
        almacen.leer_comparativa("cmp-inexistente")


# ---- diferencias ceteris paribus -------------------------------------------------------------


def _config(**cambios) -> dict:
    base = {"estrategia": "bayes_stein", "parametros": {"ventana_dias": 504, "peso_maximo": 0.5}, "universo": TICKERS,
            "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31", "frecuencia_rebalanceo": "monthly",
            "comision": 0.001, "slippage": 0.001, "benchmark": "SPY", "capital_inicial": 100_000.0, "semilla": 42,
            "dias_historia_previa": 1100}
    return {**base, **cambios}


def test_sin_diferencias_cuando_solo_cambia_la_estrategia_y_sus_parametros_propios():
    configs = {"a": _config(), "b": _config(estrategia="momentum", parametros={"top_k": 3}),
               "c": _config(parametros={"ventana_dias": 504, "peso_maximo": 0.5, "penalizacion_turnover": 2.0})}
    assert diferencias_ceteris_paribus(configs) == []


def test_detecta_diferencias_de_entorno_y_de_comunes():
    configs = {"a": _config(), "b": _config(fecha_fin="2022-12-31", comision=0.002, universo=["AAA", "BBB"],
                                            parametros={"ventana_dias": 252, "peso_maximo": 0.5})}
    diferencias = diferencias_ceteris_paribus(configs)
    assert {d["campo"] for d in diferencias} == {"fecha_fin", "comision", "universo", "ventana_dias"}
    detalle = next(d for d in diferencias if d["campo"] == "ventana_dias")
    assert detalle["valores"] == {"a": 504, "b": 252} and detalle["tipo"] == "comun"


def test_el_orden_del_universo_no_cuenta_y_un_comun_que_no_declara_una_estrategia_tampoco():
    configs = {"a": _config(universo=list(reversed(TICKERS))), "b": _config(),
               "c": _config(estrategia="momentum", parametros={"top_k": 3})}
    assert diferencias_ceteris_paribus(configs) == []


def test_experimentos_importados_del_tfm_se_comparan_por_los_campos_de_config():
    configs = {"a": _config(), "tfm": _config(estrategia="tfm_bayes_stein", parametros={}, fecha_inicio="2015-01-01")}
    assert [d["campo"] for d in diferencias_ceteris_paribus(configs)] == ["fecha_inicio"]


def test_con_un_solo_experimento_no_hay_nada_que_comparar():
    assert diferencias_ceteris_paribus({"a": _config()}) == []
