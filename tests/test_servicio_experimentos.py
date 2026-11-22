"""
Tests de `ServicioExperimentos` SIN FastAPI: lanzar, reutilizar por huella, forzar, estados, barridos, comparativas
y backtests de una configuración guardada, con precios sintéticos y un ejecutor síncrono (sin hilos).

Se relaciona con: ypyquant.estrategias.servicio_experimentos, .almacen (índice por receta), .marcos y .configuraciones;
tests/test_api*.py cubren la traducción HTTP de los mismos flujos.
"""

from __future__ import annotations

from concurrent.futures import Executor, Future
from datetime import date

import pytest

from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.barridos import AlmacenBarridos, ConfiguracionBarrido
from ypyquant.estrategias.configuraciones import ConfiguracionGuardada
from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.estrategias.marcos import AlmacenMarcos, MarcoComparacion
from ypyquant.estrategias.precios import CargadorSintetico
from ypyquant.estrategias.servicio_experimentos import Lanzamiento, ServicioExperimentos

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


class EjecutorSincrono(Executor):
    """Ejecuta cada tarea al encolarla (determinista y sin hilos); `pendientes=True` las retiene hasta `soltar()`."""

    def __init__(self, retener: bool = False):
        self.retener, self.cola = retener, []

    def submit(self, fn, /, *args, **kwargs):
        futuro: Future = Future()
        if self.retener:
            self.cola.append((futuro, fn, args, kwargs))
        else:
            futuro.set_result(fn(*args, **kwargs))
        return futuro

    def soltar(self) -> None:
        for futuro, fn, args, kwargs in self.cola:
            futuro.set_result(fn(*args, **kwargs))
        self.cola.clear()


class CargadorQueFalla:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("sin red")


def _servicio(tmp_path, ejecutor=None, cargador=None) -> ServicioExperimentos:
    almacen = AlmacenExperimentos(tmp_path / "experimentos")
    return ServicioExperimentos(almacen, cargador or CargadorSintetico(), ejecutor or EjecutorSincrono(),
                                AlmacenBarridos(tmp_path / "barridos", almacen), AlmacenMarcos(tmp_path / "marcos"))


def _config(**cambios) -> ConfiguracionExperimento:
    base = {"estrategia": "momentum", "parametros": {"top_k": 2}, "universo": TICKERS,
            "fecha_inicio": "2020-01-01", "fecha_fin": "2021-06-30"}
    return ConfiguracionExperimento(**{**base, **cambios})


def _marco(**cambios) -> MarcoComparacion:
    base = {"id": "marco-prueba", "nombre": "Marco de prueba", "universo": TICKERS, "fecha_inicio": "2020-01-01",
            "fecha_fin": "2021-06-30", "frecuencia_rebalanceo": "monthly",
            "parametros_comunes": {"ventana_dias": 200, "peso_maximo": 0.6}}
    return MarcoComparacion(**{**base, **cambios})


# ---- lanzar / reutilizar / forzar / estados ------------------------------------------------------


def test_lanzar_ejecuta_y_deja_el_experimento_hecho(tmp_path):
    servicio = _servicio(tmp_path)
    lanzamiento = servicio.lanzar(_config())
    assert isinstance(lanzamiento, Lanzamiento) and lanzamiento.reutilizado is False
    assert lanzamiento.id == _config().normalizada().identificador()
    assert servicio.estado_de(lanzamiento.id) == "hecho"
    assert servicio.almacen.leer_metricas(lanzamiento.id)["sharpe"] is not None


def test_relanzar_la_misma_configuracion_reutiliza_y_forzar_recalcula(tmp_path):
    servicio = _servicio(tmp_path)
    primero = servicio.lanzar(_config())
    segundo = servicio.lanzar(_config(etiqueta="otra etiqueta"))  # la etiqueta no entra en la huella
    assert segundo == Lanzamiento(primero.id, True)
    forzado = servicio.lanzar(_config(), forzar=True)
    assert forzado == Lanzamiento(primero.id, False)


def test_un_experimento_en_marcha_se_reutiliza_y_uno_con_error_se_relanza(tmp_path):
    ejecutor = EjecutorSincrono(retener=True)
    servicio = _servicio(tmp_path, ejecutor)
    id_experimento = servicio.lanzar(_config()).id
    assert servicio.estado_de(id_experimento) == "pendiente"
    assert servicio.lanzar(_config()).reutilizado is True  # pendiente: se reutiliza
    ejecutor.soltar()
    assert servicio.estado_de(id_experimento) == "hecho"

    roto = _servicio(tmp_path / "roto", cargador=CargadorQueFalla())
    id_roto = roto.lanzar(_config()).id
    assert roto.estado_de(id_roto) == "error" and "sin red" in (roto.almacen.leer(id_roto)["error"] or "")
    assert roto.lanzar(_config()).reutilizado is False  # un error no se reutiliza


def test_estado_de_un_inexistente_es_none_y_un_id_invalido_lanza_valueerror(tmp_path):
    servicio = _servicio(tmp_path)
    assert servicio.estado_de("momentum-0000000000") is None
    with pytest.raises(ValueError):
        servicio.estado_de("../fuera")


def test_borrar_varios_omite_los_que_siguen_en_marcha(tmp_path):
    ejecutor = EjecutorSincrono()
    servicio = _servicio(tmp_path, ejecutor)
    hecho = servicio.lanzar(_config()).id
    ejecutor.retener = True
    en_marcha = servicio.lanzar(_config(parametros={"top_k": 3})).id
    resultado = servicio.borrar_varios([hecho, hecho, en_marcha, "momentum-inexistente"])
    assert resultado == {"borrados": [hecho], "en_marcha": [en_marcha], "no_encontrados": ["momentum-inexistente"]}


# ---- barridos ---------------------------------------------------------------------------------


def test_lanzar_barrido_encola_las_combinaciones_y_se_reutiliza(tmp_path):
    servicio = _servicio(tmp_path)
    barrido = ConfiguracionBarrido(base=_config(), rejilla={"top_k": [1, 2, 3]})
    id_barrido, ids, reutilizado = servicio.lanzar_barrido(barrido)
    assert id_barrido.startswith("barrido-") and len(ids) == 3 and reutilizado is False
    assert {servicio.estado_de(i) for i in ids} == {"hecho"}
    assert servicio.lanzar_barrido(barrido) == (id_barrido, ids, True)


def test_lanzar_varios_devuelve_los_ids_en_orden(tmp_path):
    servicio = _servicio(tmp_path)
    configs = [_config(parametros={"top_k": k}) for k in (1, 2)]
    assert servicio.lanzar_varios(configs) == [c.normalizada().identificador() for c in configs]


# ---- comparativas -----------------------------------------------------------------------------


def test_lanzar_comparativa_crea_experimentos_normales_y_registra(tmp_path):
    servicio = _servicio(tmp_path)
    marco = _marco()
    entradas = [{"estrategia": "momentum", "parametros": {"top_k": 2}},
                {"estrategia": "pesos_iguales", "parametros": {}}]
    comparativa = servicio.lanzar_comparativa(marco, entradas)
    assert comparativa["id"].startswith("cmp-") and [e["reutilizado"] for e in comparativa["experimentos"]] == [False, False]
    assert all(servicio.estado_de(e["id_experimento"]) == "hecho" for e in comparativa["experimentos"])
    assert comparativa["experimentos"][0]["informe"]["no_aplicados"] == ["peso_maximo", "ventana_dias"]
    otra_vez = servicio.lanzar_comparativa(marco, entradas)
    assert otra_vez["id"] == comparativa["id"] and all(e["reutilizado"] for e in otra_vez["experimentos"])


def test_comparativa_invalida_no_lanza_nada(tmp_path):
    servicio = _servicio(tmp_path)
    with pytest.raises(ValueError, match="no_existe"):
        servicio.lanzar_comparativa(_marco(), [{"estrategia": "momentum", "parametros": {}},
                                               {"estrategia": "no_existe", "parametros": {}}])
    assert servicio.almacen.listar() == []


# ---- configuraciones guardadas -------------------------------------------------------------------


def _guardada(servicio: ServicioExperimentos, **cambios) -> ConfiguracionGuardada:
    return ConfiguracionGuardada.desde_experimento(_config(**cambios), nombre="Mi receta")


def test_backtest_de_configuracion_con_fechas_y_con_valores_por_defecto(tmp_path):
    servicio = _servicio(tmp_path)
    guardada = _guardada(servicio)
    explicito = servicio.lanzar_backtest_de_configuracion(guardada, "2020-01-01", "2021-06-30")
    assert explicito.id == _config().normalizada().identificador()  # misma huella que el experimento de origen
    por_defecto = servicio.lanzar_backtest_de_configuracion(guardada, hoy=date(2021, 7, 1))
    assert por_defecto.id != explicito.id
    config = servicio.almacen.leer_config(por_defecto.id)
    assert config["fecha_fin"] == "2021-06-30" and config["fecha_inicio"] == "2016-07-01"


def test_backtest_de_configuracion_usa_las_fechas_del_marco_enlazado(tmp_path):
    servicio = _servicio(tmp_path)
    servicio.almacen_marcos.guardar(_marco(fecha_inicio="2020-03-01", fecha_fin="2021-03-01"))
    guardada = ConfiguracionGuardada.desde_experimento(_config(), nombre="Enlazada", id_marco="marco-prueba")
    config = servicio.almacen.leer_config(servicio.lanzar_backtest_de_configuracion(guardada).id)
    assert (config["fecha_inicio"], config["fecha_fin"]) == ("2020-03-01", "2021-03-01")


def test_backtests_de_receta_usa_el_indice_y_agrupa_por_receta(tmp_path):
    servicio = _servicio(tmp_path)
    guardada = _guardada(servicio)
    a = servicio.lanzar_backtest_de_configuracion(guardada, "2020-01-01", "2021-06-30").id
    b = servicio.lanzar_backtest_de_configuracion(guardada, "2020-06-01", "2021-06-30", capital_inicial=10_000.0).id
    servicio.lanzar(_config(parametros={"top_k": 3}))  # otra receta
    assert {r["id"] for r in servicio.backtests_de_receta(guardada)} == {a, b}


def test_comparativa_de_configuracion_crea_o_reutiliza_el_marco(tmp_path):
    servicio = _servicio(tmp_path)
    guardada = _guardada(servicio)
    entradas = [{"estrategia": "momentum", "parametros": {"top_k": 2}}, {"estrategia": "pesos_iguales", "parametros": {}}]
    primera = servicio.lanzar_comparativa_de_configuracion(guardada, entradas, "2020-01-01", "2021-06-30")
    assert primera["marco"]["id_configuracion"] == guardada.id
    # la propia estrategia dentro del marco reproduce el experimento de la configuración
    assert primera["experimentos"][0]["id_experimento"] == _config().normalizada().identificador()
    segunda = servicio.lanzar_comparativa_de_configuracion(guardada, entradas, "2020-01-01", "2021-06-30")
    assert segunda["id"] == primera["id"] and len(servicio.almacen_marcos.listar()) == 1
