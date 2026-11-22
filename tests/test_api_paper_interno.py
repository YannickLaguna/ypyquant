"""
test_api_paper_interno.py — Rutas `/api/paper/interno` con TestClient, precios sintéticos, raíz temporal y reloj fijo.

Responsabilidad: crear una sesión desde un experimento terminado (y desde una configuración), avanzar con el botón
(POST /avanzar + sondeo), listar con métricas, equity coherente con el simulador, pausar/reanudar/detener, 409 al
avanzar dos veces a la vez, 404/422 y el backtest espejo (reutiliza el experimento por huella y, en modo `cierre`,
coincide con la curva real: continuación de la paridad a nivel API).

Se relaciona con: app/api/rutas_paper_interno.py, app/api/contexto.py y src/ypyquant/paper/.
"""

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import crear_app
from app.api.paper import BotControl, ServicioAlpaca
from tests._paper_utiles import reloj
from ypyquant.data.fuentes import FuenteSintetica

TICKERS = ["AAA", "BBB", "CCC", "DDD"]
CONFIG = {"estrategia": "momentum", "parametros": {"top_k": 2, "meses_retorno": 2}, "universo": TICKERS,
          "fecha_inicio": "2020-01-01", "fecha_fin": "2020-12-31", "dias_historia_previa": 400}


@pytest.fixture(autouse=True)
def sin_claves(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)


def _cliente(tmp_path, dia="2021-03-12") -> TestClient:
    app = crear_app(FuenteSintetica(), tmp_path / "exp", ServicioAlpaca(), BotControl(tmp_path / "bot"), cargar_env=False,
                    catchup_paper_al_arrancar=False, reloj_paper=reloj(dia))
    return TestClient(app)


def _esperar_experimento(cliente, id_experimento, timeout=60.0):
    limite = time.time() + timeout
    while time.time() < limite:
        detalle = cliente.get(f"/api/experimentos/{id_experimento}").json()
        if detalle["estado"] in ("hecho", "error"):
            return detalle
        time.sleep(0.1)
    raise AssertionError("El experimento no terminó a tiempo")


def _esperar_avance(cliente, timeout=60.0):
    limite = time.time() + timeout
    while time.time() < limite:
        estado = cliente.get("/api/paper/interno/avanzar/estado").json()
        if not estado["en_curso"] and estado["fin"]:
            return estado
        time.sleep(0.05)
    raise AssertionError("El avance no terminó a tiempo")


def _experimento_hecho(cliente) -> str:
    id_experimento = cliente.post("/api/experimentos", json=CONFIG).json()["id"]
    assert _esperar_experimento(cliente, id_experimento)["estado"] == "hecho"
    return id_experimento


def _crear(cliente, id_experimento, **extra):
    return cliente.post("/api/paper/interno/sesiones", json={
        "id_experimento": id_experimento, "modo_ejecucion": "cierre", "fecha_alta": "2021-03-01", **extra})


def test_lista_vacia_y_estado_de_avance_inicial(tmp_path):
    cliente = _cliente(tmp_path)
    assert cliente.get("/api/paper/interno/sesiones").json() == {"sesiones": []}
    assert cliente.get("/api/paper/interno/avanzar/estado").json()["en_curso"] is False


def test_crear_desde_experimento_avanzar_y_listar(tmp_path):
    cliente = _cliente(tmp_path)
    id_experimento = _experimento_hecho(cliente)
    respuesta = _crear(cliente, id_experimento, nombre="Momentum paper", capital_inicial=50_000, comision=0.0)
    assert respuesta.status_code == 201
    sesion = respuesta.json()
    assert sesion["id"] == "momentum-paper" and sesion["id_experimento"] == id_experimento
    assert sesion["sesion"]["receta"]["comision"] == 0.0 and sesion["sesion"]["receta"]["estrategia"] == "momentum"
    assert sesion["dias"] == 0 and sesion["dias_pendientes"] == 10  # sesiones del 1 al 12 de marzo

    assert cliente.post("/api/paper/interno/avanzar").status_code == 202
    estado = _esperar_avance(cliente)
    assert estado["error"] is None and estado["resultados"][0]["dias_procesados"] == 10

    fila = cliente.get("/api/paper/interno/sesiones").json()["sesiones"][0]
    assert fila["dias"] == 10 and fila["al_dia"] is True and fila["dias_pendientes"] == 0 and fila["estrategia"] == "momentum"
    assert fila["costes"] >= 0 and fila["ultima_fecha"] == "2021-03-12"
    equity = cliente.get("/api/paper/interno/sesiones/momentum-paper/equity").json()["equity"]
    assert len(equity) == 10 and equity[0]["fecha"] == "2021-03-01" and equity[-1]["equity"] == pytest.approx(fila["equity"])
    assert cliente.get("/api/paper/interno/sesiones/momentum-paper/operaciones").json()["operaciones"]
    assert cliente.get("/api/paper/interno/sesiones/momentum-paper/pesos").json()["pesos"]
    detalle = cliente.get("/api/paper/interno/sesiones/momentum-paper").json()
    assert detalle["posiciones"] and detalle["ejecucion_optimista"] is True and detalle["efectivo"] >= 0


def test_estados_pausar_reanudar_detener(tmp_path):
    cliente = _cliente(tmp_path)
    sesion = _crear(cliente, _experimento_hecho(cliente)).json()["id"]
    base = f"/api/paper/interno/sesiones/{sesion}"
    assert cliente.post(f"{base}/pausar").json()["estado"] == "pausada"
    assert cliente.post(f"{base}/pausar").status_code == 409
    cliente.post("/api/paper/interno/avanzar")
    assert _esperar_avance(cliente)["resultados"] == []  # una pausada no entra en el avance
    assert cliente.post(f"{base}/reanudar").json()["estado"] == "activa"
    assert cliente.post(f"{base}/detener").json()["estado"] == "detenida"
    assert cliente.post(f"{base}/reanudar").status_code == 409
    assert cliente.post(f"{base}/explotar").status_code == 422


def test_errores_404_409_y_422(tmp_path):
    cliente = _cliente(tmp_path)
    assert cliente.get("/api/paper/interno/sesiones/no-existe").status_code == 404
    assert cliente.get("/api/paper/interno/sesiones/No Valido!/equity").status_code == 404
    assert cliente.post("/api/paper/interno/sesiones/no-existe/pausar").status_code == 404
    assert cliente.post("/api/paper/interno/sesiones", json={"id_experimento": "no-existe", "modo_ejecucion": "cierre"}).status_code == 404
    assert cliente.post("/api/paper/interno/sesiones", json={"id_configuracion": "no-existe"}).status_code == 404
    assert cliente.post("/api/paper/interno/sesiones", json={}).status_code == 422  # sin origen
    assert cliente.post("/api/paper/interno/sesiones", json={"id_experimento": "a", "id_configuracion": "b"}).status_code == 422
    id_experimento = _experimento_hecho(cliente)
    assert cliente.post("/api/paper/interno/sesiones", json={"id_experimento": id_experimento, "modo_ejecucion": "intradia"}).status_code == 422
    assert cliente.post("/api/paper/interno/sesiones", json={"id_experimento": id_experimento, "capital_inicial": -5}).status_code == 422


def test_experimento_sin_terminar_da_409(tmp_path):
    cliente = _cliente(tmp_path)
    contexto = cliente.app.state.ctx
    from ypyquant.estrategias.experimentos import ConfiguracionExperimento
    id_pendiente = contexto.almacen.registrar(ConfiguracionExperimento(**CONFIG))  # registrado, sin ejecutar
    assert cliente.post("/api/paper/interno/sesiones", json={"id_experimento": id_pendiente}).status_code == 409


def test_avanzar_dos_veces_a_la_vez_da_409(tmp_path):
    cliente = _cliente(tmp_path)
    contexto = cliente.app.state.ctx
    contexto.control_avance_paper.iniciar()  # simula un avance en curso
    assert cliente.post("/api/paper/interno/avanzar").status_code == 409
    contexto.control_avance_paper.ejecutar(list)
    assert cliente.post("/api/paper/interno/avanzar").status_code == 202
    _esperar_avance(cliente)


def test_crear_desde_configuracion(tmp_path):
    cliente = _cliente(tmp_path)
    id_experimento = _experimento_hecho(cliente)
    configuracion = cliente.post("/api/configuraciones", json={"nombre": "Mi momentum", "desde_experimento": id_experimento}).json()
    respuesta = cliente.post("/api/paper/interno/sesiones", json={"id_configuracion": configuracion["id"], "modo_ejecucion": "apertura_siguiente"})
    assert respuesta.status_code == 201
    assert respuesta.json()["id_configuracion"] == "mi-momentum" and respuesta.json()["modo_ejecucion"] == "apertura_siguiente"


def test_espejo_reutiliza_el_experimento_y_coincide_en_modo_cierre(tmp_path):
    cliente = _cliente(tmp_path)
    sesion = _crear(cliente, _experimento_hecho(cliente)).json()["id"]
    url = f"/api/paper/interno/sesiones/{sesion}/espejo"
    assert cliente.get(url).json()["estado"] == "sin_datos"  # todavía no hay días
    cliente.post("/api/paper/interno/avanzar")
    _esperar_avance(cliente)
    primera = cliente.get(url).json()
    id_espejo = primera["id_experimento"]
    assert id_espejo
    assert _esperar_experimento(cliente, id_espejo)["estado"] == "hecho"
    final = cliente.get(url).json()
    assert final["id_experimento"] == id_espejo and final["estado"] == "hecho"  # misma huella: no se relanza
    real = np.array([p["equity"] for p in final["real"]])
    espejo = np.array([p["equity"] for p in final["espejo"]])
    assert [p["fecha"] for p in final["real"]] == [p["fecha"] for p in final["espejo"]]
    np.testing.assert_allclose(real, espejo, rtol=1e-9)


def test_catchup_al_arrancar_pone_al_dia_las_sesiones_activas(tmp_path):
    cliente = _cliente(tmp_path)
    sesion = _crear(cliente, _experimento_hecho(cliente)).json()["id"]
    app = crear_app(FuenteSintetica(), tmp_path / "exp", ServicioAlpaca(), BotControl(tmp_path / "bot"), cargar_env=False,
                    catchup_paper_al_arrancar=True, reloj_paper=reloj("2021-03-12"))
    nuevo = TestClient(app)
    estado = _esperar_avance(nuevo)
    assert estado["resultados"][0]["id_sesion"] == sesion and estado["resultados"][0]["dias_procesados"] == 10


def test_avances_en_hilos_distintos_no_corrompen_la_sesion(tmp_path):
    cliente = _cliente(tmp_path)
    sesion = _crear(cliente, _experimento_hecho(cliente)).json()["id"]
    servicio = cliente.app.state.ctx.servicio_paper_interno
    resultados = []
    hilos = [threading.Thread(target=lambda: resultados.append(servicio.simulador.avanzar(sesion))) for _ in range(3)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join()
    assert sum(r.dias_procesados for r in resultados) == 10  # cada día se procesó una sola vez
    assert len(servicio.almacen.leer_tablas(sesion).equity) == 10
