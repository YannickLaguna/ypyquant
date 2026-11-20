"""
test_api_configuraciones.py — Tests de las rutas `/api/configuraciones` (fase 1) con TestClient.

Usa precios sintéticos y almacenes temporales (sin red). Se relaciona con: app/api/rutas_configuraciones.py,
app/api/contexto.py, ypyquant.estrategias.configuraciones y tests/test_api.py (mismo patrón).
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.api.main import crear_app
from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.estrategias.precios import CargadorSintetico

TICKERS = ["AAA", "BBB", "CCC", "DDD"]
CONFIG = {"estrategia": "momentum", "parametros": {"top_k": 2}, "universo": TICKERS,
          "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}


@pytest.fixture(autouse=True)
def sin_claves(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)


def _cliente(tmp_path) -> TestClient:
    app = crear_app(CargadorSintetico(), tmp_path / "exp", ServicioAlpaca(), BotControl(tmp_path / "paper"), cargar_env=False)
    return TestClient(app)


def _esperar(cliente: TestClient, id_experimento: str, timeout: float = 60.0) -> dict:
    limite = time.time() + timeout
    while time.time() < limite:
        detalle = cliente.get(f"/api/experimentos/{id_experimento}").json()
        if detalle["estado"] in ("hecho", "error"):
            return detalle
        time.sleep(0.1)
    raise AssertionError("El experimento no terminó a tiempo")


def _experimento_hecho(cliente: TestClient, **cambios) -> str:
    id_experimento = cliente.post("/api/experimentos", json={**CONFIG, **cambios}).json()["id"]
    assert _esperar(cliente, id_experimento)["estado"] == "hecho"
    return id_experimento


def test_crear_desde_experimento_guarda_receta_y_metricas_de_origen(tmp_path):
    cliente = _cliente(tmp_path)
    id_experimento = _experimento_hecho(cliente)
    respuesta = cliente.post("/api/configuraciones", json={"nombre": "Momentum top 2", "desde_experimento": id_experimento})
    assert respuesta.status_code == 201
    cuerpo = respuesta.json()
    assert cuerpo["id"] == "momentum-top-2" and cuerpo["version"] == 1 and cuerpo["id_marco"] is None
    assert cuerpo["origen"]["tipo"] == "experimento" and cuerpo["origen"]["id_experimento"] == id_experimento
    assert "sharpe" in cuerpo["origen"]["metricas"] and cuerpo["origen"]["fechas"] == ["2020-01-01", "2021-12-31"]
    assert cuerpo["parametros"]["top_k"] == 2 and "fecha_inicio" not in cuerpo
    assert cliente.post("/api/configuraciones", json={"nombre": "Momentum top 2", "desde_experimento": id_experimento}).status_code == 409


def test_crear_errores_de_origen(tmp_path):
    cliente = _cliente(tmp_path)
    assert cliente.post("/api/configuraciones", json={"nombre": "x", "desde_experimento": "no-existe"}).status_code == 404
    assert cliente.post("/api/configuraciones", json={"nombre": "x"}).status_code == 422  # sin origen
    assert cliente.post("/api/configuraciones", json={"nombre": "x", "desde_experimento": "a", "desde_barrido": "b"}).status_code == 422
    assert cliente.post("/api/configuraciones", json={"nombre": "x", "desde_barrido": "barrido-0000000000"}).status_code == 404


def test_crear_desde_barrido_con_criterio_mejor(tmp_path):
    cliente = _cliente(tmp_path)
    barrido = cliente.post("/api/experimentos/barrido", json={"config": CONFIG, "rejilla": {"top_k": [1, 2, 3]}}).json()
    assert all(_esperar(cliente, i)["estado"] == "hecho" for i in barrido["ids"])
    respuesta = cliente.post("/api/configuraciones", json={
        "nombre": "Mejor del barrido", "desde_barrido": barrido["barrido_id"], "criterio": "mejor"})
    assert respuesta.status_code == 201
    origen = respuesta.json()["origen"]
    assert origen["tipo"] == "barrido" and origen["id_barrido"] == barrido["barrido_id"] and origen["criterio"] == "mejor"
    assert origen["id_experimento"] in barrido["ids"] and "sobreajust" in origen["aviso"]
    # El cliente puede elegir una combinación concreta del barrido (criterio elegida); una ajena es 422
    elegido = barrido["ids"][0]
    ok = cliente.post("/api/configuraciones", json={"nombre": "Elegida", "desde_barrido": barrido["barrido_id"],
                                                    "criterio": "elegida", "id_experimento": elegido})
    assert ok.status_code == 201 and ok.json()["origen"]["id_experimento"] == elegido
    ajeno = cliente.post("/api/configuraciones", json={"nombre": "Ajena", "desde_barrido": barrido["barrido_id"],
                                                       "criterio": "elegida", "id_experimento": "otro-123"})
    assert ajeno.status_code == 422


def test_crear_manual_y_validacion_de_parametros(tmp_path):
    cliente = _cliente(tmp_path)
    receta = {"estrategia": "momentum", "parametros": {"top_k": 3}, "universo": TICKERS}
    assert cliente.post("/api/configuraciones", json={"nombre": "Manual", "manual": receta}).status_code == 201
    mala = {**receta, "parametros": {"top_k": 0}}
    assert cliente.post("/api/configuraciones", json={"nombre": "Mala", "manual": mala}).status_code == 422
    assert cliente.post("/api/configuraciones", json={"nombre": "Sin estrategia", "manual": {**receta, "estrategia": "nada"}}).status_code == 404


def test_listar_detalle_versiones_y_archivar(tmp_path):
    cliente = _cliente(tmp_path)
    receta = {"estrategia": "momentum", "parametros": {"top_k": 3}, "universo": TICKERS}
    cliente.post("/api/configuraciones", json={"nombre": "Manual", "manual": receta})
    assert [c["id"] for c in cliente.get("/api/configuraciones").json()["configuraciones"]] == ["manual"]
    assert cliente.get("/api/configuraciones", params={"estrategia": "otra"}).json()["configuraciones"] == []

    nueva = cliente.post("/api/configuraciones/manual/versiones", json={"cambios": {"parametros": {"top_k": 4}}})
    assert nueva.status_code == 201 and nueva.json()["version"] == 2
    assert cliente.post("/api/configuraciones/manual/versiones", json={"cambios": {"parametros": {"top_k": 0}}}).status_code == 422
    assert cliente.post("/api/configuraciones/manual/versiones", json={"cambios": {"estrategia": "pesos_iguales"}}).status_code == 422

    detalle = cliente.get("/api/configuraciones/manual").json()
    assert detalle["configuracion"]["version"] == 2 and [v["version"] for v in detalle["versiones"]] == [1, 2]
    assert cliente.get("/api/configuraciones/manual", params={"version": 1}).json()["configuracion"]["parametros"]["top_k"] == 3
    assert cliente.get("/api/configuraciones/nada").status_code == 404

    assert cliente.delete("/api/configuraciones/manual").status_code == 204
    assert cliente.get("/api/configuraciones").json()["configuraciones"] == []
    assert len(cliente.get("/api/configuraciones", params={"archivadas": True}).json()["configuraciones"]) == 1
    assert cliente.delete("/api/configuraciones/manual", params={"definitivo": True}).status_code == 204
    assert cliente.delete("/api/configuraciones/manual").status_code == 404


def test_backtest_desde_configuracion_reutiliza_el_experimento_de_origen(tmp_path):
    cliente = _cliente(tmp_path)
    id_experimento = _experimento_hecho(cliente)
    cliente.post("/api/configuraciones", json={"nombre": "Base", "desde_experimento": id_experimento})
    mismas_fechas = cliente.post("/api/configuraciones/base/backtest", json={"fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"})
    assert mismas_fechas.status_code == 202
    assert mismas_fechas.json() == {"id_experimento": id_experimento, "reutilizado": True}
    otras = cliente.post("/api/configuraciones/base/backtest", json={"fecha_inicio": "2020-06-01", "fecha_fin": "2021-06-30"})
    assert otras.json()["reutilizado"] is False and otras.json()["id_experimento"] != id_experimento
    assert _esperar(cliente, otras.json()["id_experimento"])["estado"] == "hecho"
    # Sin fechas se usan las de los últimos 5 años; fechas incoherentes son 422
    assert cliente.post("/api/configuraciones/base/backtest", json={"fecha_inicio": "2022-01-01", "fecha_fin": "2021-01-01"}).status_code == 422
    assert cliente.post("/api/configuraciones/nada/backtest", json={}).status_code == 404


def test_detalle_lista_los_backtests_asociados_a_la_receta(tmp_path):
    cliente = _cliente(tmp_path)
    id_experimento = _experimento_hecho(cliente)
    cliente.post("/api/configuraciones", json={"nombre": "Base", "desde_experimento": id_experimento})
    otro = cliente.post("/api/configuraciones/base/backtest", json={"fecha_inicio": "2020-06-01", "fecha_fin": "2021-06-30"}).json()
    _esperar(cliente, otro["id_experimento"])
    _experimento_hecho(cliente, parametros={"top_k": 3})  # otra receta: no debe aparecer
    asociados = {e["id"] for e in cliente.get("/api/configuraciones/base").json()["backtests"]}
    assert asociados == {id_experimento, otro["id_experimento"]}
