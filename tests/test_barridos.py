"""
Tests de barridos y su validación: matriz de retornos, superficie con meseta, validación (DSR, PBO,
rachas), permutación (serie y con procesos), walk-forward, periodo reservado con contador y la API
`/api/barridos` (TestClient, almacén temporal, precios sintéticos, sin red).

Se relaciona con: ypyquant.estrategias.{barridos, validacion_barridos, reservado},
app/api/rutas_barridos.py y tests/test_estrategias.py (misma configuración base).
"""

import math
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.main import crear_app
from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.estrategias import registro
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.barridos import (
    AlmacenBarridos,
    ConfiguracionBarrido,
    _vecinos,
    matriz_retornos,
    superficie,
)
from ypyquant.estrategias.base import Estrategia, ParametrosBase
from ypyquant.estrategias.experimentos import (
    ConfiguracionExperimento,
    ejecutar_experimento,
    expandir_rejilla,
    expandir_rejilla_con_omitidas,
)
from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.reservado import RegistroReservado, evaluar_reservado
from ypyquant.estrategias.validacion_barridos import (
    BarridoEnCurso,
    exigir_terminado,
    prueba_permutacion_barrido,
    validar_barrido,
    walk_forward_barrido,
)
from ypyquant.validacion import seleccion_walk_forward, sharpe

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
BASE = {"estrategia": "momentum", "universo": TICKERS, "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}
REJILLA = {"top_k": [1, 2, 3], "meses_retorno": [3, 6, 9]}
FIN_DESARROLLO = "2021-06-30"


def _barrido_config(**cambios) -> ConfiguracionBarrido:
    datos = {"base": ConfiguracionExperimento(**BASE), "rejilla": REJILLA, "fecha_fin_desarrollo": FIN_DESARROLLO}
    return ConfiguracionBarrido(**{**datos, **cambios})


def _ejecutar_todo(almacen: AlmacenBarridos, config: ConfiguracionBarrido):
    """Crea el barrido y ejecuta en serie todas sus combinaciones (lo que hace la API en segundo plano)."""
    id_barrido, _ = almacen.crear(config)
    for combinacion_config, _, _ in config.combinaciones():
        id_exp = almacen.experimentos.registrar(combinacion_config)
        almacen.experimentos.guardar(id_exp, ejecutar_experimento(combinacion_config, FuenteSintetica()))
    return almacen.cargar(id_barrido)


@pytest.fixture(scope="module")
def almacen(tmp_path_factory) -> AlmacenBarridos:
    raiz = tmp_path_factory.mktemp("barridos")
    return AlmacenBarridos(raiz / "barridos", AlmacenExperimentos(raiz / "experimentos"))


@pytest.fixture(scope="module")
def barrido(almacen):
    """Barrido 3x3 de momentum con periodo reservado (2021-07-01 a 2021-12-31), ya ejecutado."""
    return _ejecutar_todo(almacen, _barrido_config())


# ---- configuración y almacén --------------------------------------------------------


def test_configuracion_barrido_huella_y_validaciones():
    a, b = _barrido_config(), _barrido_config(etiqueta="otra nota")
    assert a.huella() == b.huella() and a.identificador().startswith("barrido-") and len(a.identificador()) == 18
    assert a.huella() != _barrido_config(fecha_fin_desarrollo=None).huella()
    assert all(c.fecha_fin == FIN_DESARROLLO for c, _, _ in a.combinaciones()) and len(a.combinaciones()) == 9
    with pytest.raises(ValueError, match="metrica_objetivo"):
        _barrido_config(metrica_objetivo="volatilidad")
    with pytest.raises(ValueError, match="fecha_fin_desarrollo"):
        _barrido_config(fecha_fin_desarrollo="2021-12-31")
    with pytest.raises(ValueError, match="al menos un eje"):
        _barrido_config(rejilla={})


def test_almacen_barridos_estado_progreso_y_borrado(tmp_path):
    almacen = AlmacenBarridos(tmp_path / "b", AlmacenExperimentos(tmp_path / "e"))
    config = _barrido_config(rejilla={"top_k": [1, 2]})
    id_barrido, creado = almacen.crear(config)
    assert creado and almacen.crear(config) == (id_barrido, False)
    assert almacen.leer(id_barrido)["estado"] == "error"  # experimentos aún inexistentes
    ids = [c["id"] for c in almacen.leer(id_barrido)["combinaciones"]]
    for i, (cfg, _, _) in enumerate(config.combinaciones()):
        almacen.experimentos.registrar(cfg)
    assert almacen.leer(id_barrido)["estado"] == "pendiente"
    almacen.experimentos.marcar_estado(ids[0], "en_curso")
    assert almacen.leer(id_barrido)["estado"] == "en_curso"
    with pytest.raises(BarridoEnCurso):
        exigir_terminado(almacen.cargar(id_barrido))
    assert almacen.listar()[0]["id"] == id_barrido
    assert almacen.borrar(id_barrido, con_experimentos=True) and not almacen.experimentos.existe(ids[0])
    assert not almacen.borrar(id_barrido)


# ---- matriz de retornos ---------------------------------------------------------------


def test_matriz_retornos_alineada_con_los_equity(barrido):
    matriz = matriz_retornos(barrido)
    assert list(matriz.columns) == barrido.ids and matriz.shape[1] == 9
    assert matriz.index.is_monotonic_increasing and not matriz.isna().any().any()
    assert matriz.index[-1] <= np.datetime64(FIN_DESARROLLO)  # el reservado no entra
    for id_exp in barrido.ids:
        equity = barrido.almacen.experimentos.leer_equity(id_exp).set_index("fecha")["retorno"]
        assert np.allclose(matriz[id_exp].to_numpy(), equity.loc[matriz.index].to_numpy())
    assert set(barrido.etiquetas) == set(barrido.ids) and "top_k=1" in barrido.etiquetas[barrido.ids[0]]


def test_matriz_con_fechas_de_inicio_distintas_recorta_o_rellena(tmp_path):
    almacen = AlmacenBarridos(tmp_path / "b", AlmacenExperimentos(tmp_path / "e"))
    config = _barrido_config(rejilla={"fecha_inicio": ["2020-01-01", "2020-04-01"]}, fecha_fin_desarrollo=None)
    barrido = _ejecutar_todo(almacen, config)
    recortada, rellena = matriz_retornos(barrido), matriz_retornos(barrido, relleno="cero")
    assert recortada.index[0] >= np.datetime64("2020-04-01") and not recortada.isna().any().any()
    assert rellena.index[0] < recortada.index[0] and len(rellena) > len(recortada)
    assert (rellena.iloc[:5, 1] == 0.0).all()  # la combinación que empieza tarde está en caja (0) al principio


# ---- superficie y meseta --------------------------------------------------------------


def test_superficie_3x3_y_meseta_son_la_media_de_los_vecinos(barrido):
    tabla = superficie(barrido)
    assert len(tabla) == 9 and {"eje_top_k", "eje_meses_retorno", "objetivo", "meseta", "n_vecinos", "sharpe"} <= set(tabla.columns)
    cuadricula = tabla.pivot(index="eje_top_k", columns="eje_meses_retorno", values="objetivo")
    assert cuadricula.shape == (3, 3)
    n_vecinos = tabla.pivot(index="eje_top_k", columns="eje_meses_retorno", values="n_vecinos")
    assert n_vecinos.loc[1, 3] == 2 and n_vecinos.loc[1, 6] == 3 and n_vecinos.loc[2, 6] == 4  # esquina, borde, centro
    meseta = tabla.pivot(index="eje_top_k", columns="eje_meses_retorno", values="meseta")
    esperado_centro = np.mean([cuadricula.loc[1, 6], cuadricula.loc[3, 6], cuadricula.loc[2, 3], cuadricula.loc[2, 9]])
    assert meseta.loc[2, 6] == pytest.approx(esperado_centro)
    esperado_esquina = np.mean([cuadricula.loc[1, 6], cuadricula.loc[2, 3]])
    assert meseta.loc[1, 3] == pytest.approx(esperado_esquina)  # la propia combinación no cuenta


# ---- validación -----------------------------------------------------------------------


def test_validar_barrido_claves_y_rangos(barrido):
    r = validar_barrido(barrido)
    assert r["n_combinaciones"] == 9 and r["metrica_objetivo"] == "sharpe"
    mejor = max(barrido.almacen.experimentos.leer_metricas(i)["sharpe"] for i in barrido.ids)
    assert r["elegida"]["objetivo"] == pytest.approx(mejor)
    assert 0.0 <= r["dsr"]["dsr"] <= 1.0 and r["dsr"]["n_pruebas"] == 9 and r["dsr"]["sr_umbral"] > 0
    assert 0.0 <= r["pbo"]["pbo"] <= 1.0 and r["pbo"]["n_bloques"] == 16 and len(r["pbo"]["pares_is_oos"]) > 0
    assert 0.0 <= r["rachas"]["p_valor"] <= 1.0 and len(r["superficie"]) == 9
    assert r["alternativa_meseta"]["id"] in barrido.ids and r["avisos"] == []
    assert barrido.almacen.leer_resultado(barrido.id, "validacion")["dsr"]["n_pruebas"] == 9


def test_validar_con_pocas_observaciones_reduce_bloques_y_avisa(barrido):
    r = validar_barrido(barrido, n_bloques=60)
    assert r["pbo"] is not None and r["pbo"]["n_bloques"] < 60 and any("bloques" in a for a in r["avisos"])


# ---- walk-forward ---------------------------------------------------------------------


def test_walk_forward_barrido_sin_lookahead(barrido):
    r = walk_forward_barrido(barrido, ventana_entreno_dias=180, paso_dias=60)
    matriz = matriz_retornos(barrido)
    ventana, paso = r["ventana_filas"], r["paso_filas"]
    assert r["elecciones"][0]["fecha"] == matriz.index[ventana].strftime("%Y-%m-%d")  # el primer uso es tras el entreno
    assert r["curva"]["fechas"][0] == r["elecciones"][0]["fecha"]
    for k, eleccion in enumerate(r["elecciones"]):
        t = ventana + k * paso
        entreno = matriz.iloc[t - ventana:t]  # solo filas anteriores a t
        assert eleccion["metrica_entreno"] == pytest.approx(float(np.nanmax(sharpe(entreno.to_numpy()))))
        assert eleccion["id"] == entreno.columns[int(np.nanargmax(sharpe(entreno.to_numpy())))]
    # cambiar el futuro (todo lo posterior al primer tramo) no altera la primera elección
    alterada = matriz.copy()
    alterada.iloc[ventana + paso:] = 0.123
    _, elecciones_alteradas, _ = seleccion_walk_forward(alterada, ventana, paso)
    assert elecciones_alteradas["columna"].iloc[0] == r["elecciones"][0]["id"]
    assert set(r["metricas"]) >= {"sharpe_anualizado", "cagr", "max_drawdown"}
    assert barrido.almacen.documentos.tiene_parquet(barrido.id, "walk_forward_oos")
    with pytest.raises(ValueError, match="no cabe"):
        walk_forward_barrido(barrido, ventana_entreno_dias=5000, paso_dias=30)


# ---- permutación ----------------------------------------------------------------------


def test_permutacion_en_serie_in_sample(barrido):
    avances = []
    r = prueba_permutacion_barrido(barrido, FuenteSintetica(), n_permutaciones=3, semilla=7,
                                   callback_progreso=lambda hechas, total: avances.append((hechas, total)))
    assert avances == [(1, 3), (2, 3), (3, 3)]
    assert r["modo"] == "in_sample" and r["n_validas"] == 3 and len(r["distribucion"]) == 3 and r["n_backtests"] == 27
    assert 0.25 <= r["p_valor"] <= 1.0 and r["valor_real"] == pytest.approx(
        max(barrido.almacen.experimentos.leer_metricas(i)["sharpe"] for i in barrido.ids))
    assert barrido.almacen.leer_resultado(barrido.id, "permutacion_in_sample")["p_valor"] == r["p_valor"]
    otra = prueba_permutacion_barrido(barrido, FuenteSintetica(), n_permutaciones=3, semilla=7)
    assert np.allclose(otra["distribucion"], r["distribucion"])  # misma semilla, misma distribución


def test_permutacion_walk_forward_y_limites(barrido):
    r = prueba_permutacion_barrido(barrido, FuenteSintetica(), n_permutaciones=2, modo="walk_forward",
                                   ventana_entreno_dias=180, paso_dias=60)
    assert r["estadistico"] == "sharpe_periodo_walk_forward" and r["inicio_permutacion"] == "2020-06-29"
    assert 0.0 < r["p_valor"] <= 1.0 and math.isfinite(r["valor_real"])
    with pytest.raises(ValueError, match="1000"):
        prueba_permutacion_barrido(barrido, FuenteSintetica(), n_permutaciones=1001)
    with pytest.raises(ValueError, match="walk_forward necesita"):
        prueba_permutacion_barrido(barrido, FuenteSintetica(), n_permutaciones=2, modo="walk_forward")


def test_permutacion_cancelacion_cooperativa(barrido):
    import threading

    from ypyquant.estrategias.validacion_barridos import PruebaCancelada

    evento = threading.Event()
    with pytest.raises(PruebaCancelada):
        prueba_permutacion_barrido(barrido, FuenteSintetica(), n_permutaciones=5,
                                   callback_progreso=lambda h, t: evento.set(), cancelar=evento)


@pytest.mark.lenta
def test_permutacion_con_procesos_da_lo_mismo_que_en_serie(barrido):
    serie = prueba_permutacion_barrido(barrido, FuenteSintetica(), n_permutaciones=4, semilla=3, procesos=1)
    paralela = prueba_permutacion_barrido(barrido, FuenteSintetica(), n_permutaciones=4, semilla=3, procesos=2)
    assert np.allclose(paralela["distribucion"], serie["distribucion"]) and paralela["p_valor"] == serie["p_valor"]


# ---- periodo reservado ----------------------------------------------------------------


def test_reservado_cuenta_usos_y_avisa_en_el_segundo(tmp_path, barrido):
    registro = RegistroReservado(tmp_path / "reservado.json")
    primero = evaluar_reservado(barrido, barrido.etiquetas[barrido.ids[0]], FuenteSintetica(), registro)
    assert primero["usos_previos"] == [] and primero["aviso"] is None and primero["n_usos"] == 1
    assert primero["periodo"] == {"inicio": "2021-07-01", "fin": "2021-12-31"} and primero["metricas"]["sharpe"] is not None
    reservado = barrido.almacen.experimentos.leer(primero["experimento_reservado"])
    assert reservado["equity"][0]["fecha"] >= "2021-07-01" and reservado["procedencia"]["hash_precios"]

    segundo = evaluar_reservado(barrido, barrido.ids[4], FuenteSintetica(), registro)  # otra combinación, mismo periodo
    assert len(segundo["usos_previos"]) == 1 and "ya se había usado 1 vez" in segundo["aviso"]
    assert segundo["usos_previos"][0]["experimento"] == barrido.ids[0] and segundo["usos_previos"][0]["barrido"] == barrido.id
    assert len(registro.ruta.read_text(encoding="utf-8")) > 0
    assert len(barrido.almacen.leer_resultado(barrido.id, "reservado")) >= 2

    with pytest.raises(ValueError, match="ni un id ni una etiqueta"):
        evaluar_reservado(barrido, "no-existe", FuenteSintetica(), registro)


def test_reservado_exige_fecha_fin_desarrollo(tmp_path):
    almacen = AlmacenBarridos(tmp_path / "b", AlmacenExperimentos(tmp_path / "e"))
    sin = _ejecutar_todo(almacen, _barrido_config(rejilla={"top_k": [1, 2]}, fecha_fin_desarrollo=None))
    with pytest.raises(ValueError, match="fecha_fin_desarrollo"):
        evaluar_reservado(sin, sin.ids[0], FuenteSintetica(), RegistroReservado(tmp_path / "r.json"))


# ---- API ------------------------------------------------------------------------------


def _cliente(tmp_path) -> TestClient:
    app = crear_app(FuenteSintetica(), tmp_path / "exp", ServicioAlpaca(), BotControl(tmp_path / "paper"),
                    cargar_env=False, raiz_barridos=tmp_path / "barridos")
    return TestClient(app)


def _esperar_barrido(cliente: TestClient, id_barrido: str, timeout: float = 120.0) -> dict:
    limite = time.time() + timeout
    while time.time() < limite:
        detalle = cliente.get(f"/api/barridos/{id_barrido}").json()
        if detalle["estado"] in ("hecho", "error"):
            return detalle
        time.sleep(0.2)
    raise AssertionError("El barrido no terminó a tiempo")


def _cuerpo_barrido(**cambios) -> dict:
    return {"base": BASE, "rejilla": {"top_k": [1, 2], "meses_retorno": [3, 6]}, "fecha_fin_desarrollo": FIN_DESARROLLO,
            **cambios}


def test_api_barrido_ciclo_completo(tmp_path):
    cliente = _cliente(tmp_path)
    respuesta = cliente.post("/api/barridos", json=_cuerpo_barrido(etiqueta="demo"))
    assert respuesta.status_code == 202
    cuerpo = respuesta.json()
    id_barrido = cuerpo["id"]
    assert id_barrido.startswith("barrido-") and cuerpo["total"] == 4 and len(cuerpo["ids"]) == 4 and cuerpo["reutilizado"] is False

    detalle = _esperar_barrido(cliente, id_barrido)
    assert detalle["estado"] == "hecho" and detalle["progreso"]["hechos"] == 4
    assert len(detalle["superficie"]) == 4 and detalle["ejes"] == {"top_k": [1, 2], "meses_retorno": [3, 6]}
    assert detalle["validacion"] is None and detalle["permutacion_in_sample"] is None and detalle["reservado"] == []
    assert cliente.get("/api/barridos").json()["barridos"][0]["id"] == id_barrido

    validacion = cliente.post(f"/api/barridos/{id_barrido}/validar", json={"n_bloques": 8})
    assert validacion.status_code == 200 and validacion.json()["pbo"]["n_bloques"] == 8
    assert cliente.get(f"/api/barridos/{id_barrido}").json()["validacion"]["dsr"]["n_pruebas"] == 4

    wf = cliente.post(f"/api/barridos/{id_barrido}/walk_forward", json={"ventana_entreno_dias": 150, "paso_dias": 60})
    assert wf.status_code == 200 and wf.json()["metricas"]["sharpe_anualizado"] is not None

    permutacion = cliente.post(f"/api/barridos/{id_barrido}/permutacion", json={"n_permutaciones": 2, "procesos": 1})
    assert permutacion.status_code == 202 and permutacion.json()["tarea"] == "permutacion_in_sample"
    limite = time.time() + 120
    while time.time() < limite:
        tarea = cliente.get(f"/api/barridos/{id_barrido}").json()["tareas"]["permutacion_in_sample"]
        if tarea["estado"] in ("hecho", "error"):
            break
        time.sleep(0.2)
    assert tarea["estado"] == "hecho", tarea
    completo = cliente.get(f"/api/barridos/{id_barrido}").json()
    assert 0 < completo["permutacion_in_sample"]["p_valor"] <= 1 and tarea["progreso"] == {"completadas": 2, "total": 2}

    r1 = cliente.post(f"/api/barridos/{id_barrido}/reservado", json={"experimento": detalle["combinaciones"][0]["id"]})
    assert r1.status_code == 200 and r1.json()["aviso"] is None
    r2 = cliente.post(f"/api/barridos/{id_barrido}/reservado", json={"experimento": detalle["combinaciones"][1]["etiqueta"]})
    assert "ya se había usado" in r2.json()["aviso"] and len(r2.json()["usos_previos"]) == 1

    # los experimentos individuales exponen su procedencia
    experimento = cliente.get(f"/api/experimentos/{detalle['combinaciones'][0]['id']}").json()
    assert experimento["procedencia"]["hash_precios"]

    assert cliente.delete(f"/api/barridos/{id_barrido}").status_code == 204
    assert cliente.get(f"/api/experimentos/{detalle['combinaciones'][0]['id']}").status_code == 200  # no se borran
    assert cliente.get(f"/api/barridos/{id_barrido}").status_code == 404
    assert cliente.delete(f"/api/barridos/{id_barrido}").status_code == 404


def test_api_409_si_hay_experimentos_en_curso_y_422_con_mensajes(tmp_path):
    cliente = _cliente(tmp_path)
    ctx = cliente.app.state.ctx
    ctx.ejecutor.submit = lambda *a, **k: None  # los experimentos se quedan en `pendiente`
    id_barrido = cliente.post("/api/barridos", json=_cuerpo_barrido()).json()["id"]
    for ruta, cuerpo in (("validar", {}), ("walk_forward", {"ventana_entreno_dias": 100, "paso_dias": 30}),
                         ("permutacion", {"n_permutaciones": 2}), ("reservado", {"experimento": "x"})):
        respuesta = cliente.post(f"/api/barridos/{id_barrido}/{ruta}", json=cuerpo)
        assert respuesta.status_code == 409, ruta
        assert "sin terminar" in respuesta.json()["detail"]

    assert cliente.post("/api/barridos", json=_cuerpo_barrido(metrica_objetivo="volatilidad")).status_code == 422
    assert cliente.post("/api/barridos", json=_cuerpo_barrido(rejilla={})).status_code == 422
    assert cliente.post("/api/barridos", json=_cuerpo_barrido(rejilla={"top_k": [0]})).status_code == 422
    assert cliente.post("/api/barridos", json=_cuerpo_barrido(base={**BASE, "estrategia": "nada"})).status_code == 404
    assert cliente.post(f"/api/barridos/{id_barrido}/permutacion", json={"n_permutaciones": 5000}).status_code == 422
    assert cliente.post("/api/barridos/no-existe/validar", json={}).status_code == 404
    assert cliente.get("/api/barridos/no-existe").status_code == 404


def test_api_experimentos_barrido_sigue_compatible_y_devuelve_el_id_del_barrido(tmp_path):
    cliente = _cliente(tmp_path)
    cuerpo = {"config": {**BASE, "estrategia": "momentum"}, "rejilla": {"top_k": [1, 2, 3]}}
    respuesta = cliente.post("/api/experimentos/barrido", json=cuerpo)
    assert respuesta.status_code == 202 and respuesta.json()["total"] == 3
    ids = respuesta.json()["ids"]
    id_barrido = respuesta.json()["barrido_id"]
    detalle = _esperar_barrido(cliente, id_barrido)
    assert [c["id"] for c in detalle["combinaciones"]] == ids and detalle["estado"] == "hecho"
    # sin rejilla (una sola combinación) no hay barrido que crear, pero la respuesta sigue siendo válida
    una = cliente.post("/api/experimentos/barrido", json={"config": BASE, "rejilla": {}}).json()
    assert una["total"] == 1 and una["barrido_id"] is None


def test_api_lista_validadores(tmp_path):
    assert {"rachas", "psr"} <= set(_cliente(tmp_path).get("/api/validadores").json()["validadores"])


# ---- rejilla tolerante a combinaciones inválidas (fase 0.2) ------------------------------


@pytest.fixture
def estrategia_cruce():
    """Estrategia temporal cuyos parámetros exigen `rapida < lenta` (hay combinaciones inválidas en la rejilla)."""
    from pydantic import model_validator

    class Cruce(Estrategia):
        nombre = "cruce_temporal"
        descripcion = "Solo para tests de rejilla tolerante."

        class Parametros(ParametrosBase):
            rapida: int = 5
            lenta: int = 20

            @model_validator(mode="after")
            def _orden(self):
                if self.rapida >= self.lenta:
                    raise ValueError("rapida debe ser menor que lenta")
                return self

        def generar_pesos(self, fecha, datos, estado):
            return {t: 1 / len(datos) for t in datos}

    antes = dict(registro._REGISTRO)
    registro.registrar_estrategia(Cruce)
    yield Cruce
    registro._REGISTRO.clear()
    registro._REGISTRO.update(antes)


REJILLA_CON_HUECO = {"rapida": [5, 50], "lenta": [20, 60]}  # (50, 20) viola rapida < lenta


def _base_cruce() -> ConfiguracionExperimento:
    return ConfiguracionExperimento(estrategia="cruce_temporal", universo=TICKERS,
                                    fecha_inicio="2020-01-01", fecha_fin="2021-12-31")


def test_expandir_rejilla_con_omitidas_devuelve_las_validas_y_registra_las_omitidas(estrategia_cruce):
    configs, omitidas = expandir_rejilla_con_omitidas(_base_cruce(), REJILLA_CON_HUECO)
    assert {(c.parametros["rapida"], c.parametros["lenta"]) for c in configs} == {(5, 20), (5, 60), (50, 60)}
    assert len(omitidas) == 1
    assert omitidas[0]["valores"] == {"rapida": 50, "lenta": 20}
    assert "rapida debe ser menor que lenta" in omitidas[0]["motivo"]


def test_expandir_rejilla_estricta_conserva_su_firma_y_sigue_fallando(estrategia_cruce):
    with pytest.raises(ValidationError):
        expandir_rejilla(_base_cruce(), REJILLA_CON_HUECO)


def test_expandir_rejilla_con_omitidas_sin_huecos_equivale_a_la_estricta():
    base = ConfiguracionExperimento(**BASE)
    configs, omitidas = expandir_rejilla_con_omitidas(base, {"top_k": [1, 2], "meses_retorno": [3, 6]})
    assert omitidas == []
    assert [c.huella() for c in configs] == [c.huella() for c in expandir_rejilla(base, {"top_k": [1, 2], "meses_retorno": [3, 6]})]


def test_barrido_con_huecos_omite_la_combinacion_y_conserva_las_posiciones(estrategia_cruce):
    config = ConfiguracionBarrido(base=_base_cruce(), rejilla=REJILLA_CON_HUECO)
    combinaciones = config.combinaciones()
    assert len(combinaciones) == 3
    assert {tuple(pos.values()) for _, _, pos in combinaciones} == {(0, 0), (0, 1), (1, 1)}  # (1, 0) es el hueco
    assert [o["valores"] for o in config.omitidas()] == [{"rapida": 50, "lenta": 20}]


def test_vecinos_toleran_posiciones_sin_valor():
    por_posicion = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}  # falta (1, 0)
    assert sorted(_vecinos({"a": 1, "b": 1}, por_posicion, ["a", "b"])) == [2.0]
    assert sorted(_vecinos({"a": 0, "b": 0}, por_posicion, ["a", "b"])) == [2.0]


def test_api_barrido_con_huecos_lanza_las_validas_y_avisa_de_las_omitidas(tmp_path, estrategia_cruce):
    cliente = _cliente(tmp_path)
    cuerpo = {"base": {**BASE, "estrategia": "cruce_temporal"}, "rejilla": REJILLA_CON_HUECO}
    respuesta = cliente.post("/api/barridos", json=cuerpo)
    assert respuesta.status_code == 202, respuesta.text
    datos = respuesta.json()
    assert datos["total"] == 3 and len(datos["omitidas"]) == 1
    detalle = _esperar_barrido(cliente, datos["id"])
    assert len(detalle["combinaciones"]) == 3 and len(detalle["omitidas"]) == 1
    # Todas inválidas: no hay nada que lanzar
    todas = cliente.post("/api/barridos", json={"base": cuerpo["base"], "rejilla": {"rapida": [50], "lenta": [20]}})
    assert todas.status_code == 422
