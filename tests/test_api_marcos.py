"""
test_api_marcos.py — Tests de las rutas de marcos y comparativas (`/api/marcos`, `/api/comparativas`) y del aviso
«no es ceteris paribus» de `GET /api/comparar`, con TestClient y precios sintéticos.

Se relaciona con: app/api/rutas_marcos.py, app/api/rutas_experimentos.py, ypyquant.estrategias.marcos y
tests/test_api_configuraciones.py (mismo patrón).
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.api.main import crear_app
from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.data.fuentes import FuenteSintetica

TICKERS = ["AAA", "BBB", "CCC", "DDD"]
MARCO = {"nombre": "Marco sintético", "universo": TICKERS, "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31",
         "frecuencia_rebalanceo": "monthly", "parametros_comunes": {"ventana_dias": 200, "peso_maximo": 0.6}}


@pytest.fixture(autouse=True)
def sin_claves(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)


def _cliente(tmp_path) -> TestClient:
    app = crear_app(FuenteSintetica(), tmp_path / "exp", ServicioAlpaca(), BotControl(tmp_path / "paper"), cargar_env=False)
    return TestClient(app)


def _esperar(cliente: TestClient, id_experimento: str, timeout: float = 90.0) -> dict:
    limite = time.time() + timeout
    while time.time() < limite:
        detalle = cliente.get(f"/api/experimentos/{id_experimento}").json()
        if detalle["estado"] in ("hecho", "error"):
            return detalle
        time.sleep(0.1)
    raise AssertionError("El experimento no terminó a tiempo")


def test_crud_de_marcos(tmp_path):
    cliente = _cliente(tmp_path)
    creado = cliente.post("/api/marcos", json=MARCO)
    assert creado.status_code == 201 and creado.json()["id"] == "marco-sintetico"
    assert cliente.post("/api/marcos", json=MARCO).status_code == 409
    assert cliente.post("/api/marcos", json={**MARCO, "nombre": "Malo", "fecha_inicio": "2023-01-01"}).status_code == 422
    assert [m["id"] for m in cliente.get("/api/marcos").json()["marcos"]] == ["marco-sintetico"]
    cambiado = cliente.put("/api/marcos/marco-sintetico", json={**MARCO, "comision": 0.002})
    assert cambiado.status_code == 200 and cliente.get("/api/marcos/marco-sintetico").json()["marco"]["comision"] == 0.002
    assert cliente.put("/api/marcos/otro", json=MARCO).status_code == 404
    assert cliente.get("/api/marcos/nada").status_code == 404
    assert cliente.delete("/api/marcos/marco-sintetico").status_code == 204
    assert cliente.delete("/api/marcos/marco-sintetico").status_code == 404


def test_lanzar_comparativa_crea_experimentos_normales_y_el_registro(tmp_path):
    cliente = _cliente(tmp_path)
    cliente.post("/api/marcos", json=MARCO)
    cuerpo = {"estrategias": [{"estrategia": "pesos_iguales"}, {"estrategia": "momentum", "parametros": {"top_k": 2}},
                              {"estrategia": "bayes_stein", "parametros": {"frecuencia_estimacion": "mensual"}}]}
    respuesta = cliente.post("/api/marcos/marco-sintetico/comparativa", json=cuerpo)
    assert respuesta.status_code == 202
    lanzada = respuesta.json()
    assert len(lanzada["experimentos"]) == 3 and lanzada["id"].startswith("cmp-")
    assert [e["estrategia"] for e in lanzada["experimentos"]] == ["pesos_iguales", "momentum", "bayes_stein"]
    informe_momentum = lanzada["experimentos"][1]["informe"]
    assert sorted(informe_momentum["no_aplicados"]) == ["peso_maximo", "ventana_dias"]
    assert lanzada["experimentos"][2]["informe"]["aplicados"] == {"ventana_dias": 200, "peso_maximo": 0.6}
    ids = [e["id_experimento"] for e in lanzada["experimentos"]]
    for id_experimento in ids:
        assert _esperar(cliente, id_experimento)["estado"] == "hecho"
    # Son experimentos normales con el entorno del marco
    detalle = cliente.get(f"/api/experimentos/{ids[2]}").json()
    assert detalle["config"]["fecha_fin"] == "2021-12-31" and detalle["config"]["parametros"]["frecuencia_estimacion"] == "mensual"

    # Relanzar lo mismo reutiliza los experimentos y la comparativa
    otra = cliente.post("/api/marcos/marco-sintetico/comparativa", json=cuerpo).json()
    assert otra["id"] == lanzada["id"] and all(e["reutilizado"] for e in otra["experimentos"])

    comparativa = cliente.get(f"/api/comparativas/{lanzada['id']}").json()
    assert comparativa["id_marco"] == "marco-sintetico" and [e["estado"] for e in comparativa["experimentos"]] == ["hecho"] * 3
    assert [c["id"] for c in cliente.get("/api/marcos/marco-sintetico").json()["comparativas"]] == [lanzada["id"]]
    assert cliente.get("/api/comparativas/cmp-0000000000").status_code == 404

    # Y comparar con el marco sale ceteris paribus
    comparacion = cliente.get("/api/comparar", params={"ids": ",".join(ids)}).json()
    assert comparacion["ceteris_paribus"] == {"ok": True, "diferencias": []}


def test_lanzar_comparativa_valida_todo_antes_de_lanzar(tmp_path):
    cliente = _cliente(tmp_path)
    cliente.post("/api/marcos", json=MARCO)
    mala = {"estrategias": [{"estrategia": "pesos_iguales"}, {"estrategia": "momentum", "parametros": {"top_k": 0}}]}
    respuesta = cliente.post("/api/marcos/marco-sintetico/comparativa", json=mala)
    assert respuesta.status_code == 422 and "momentum" in respuesta.json()["detail"]
    assert cliente.get("/api/experimentos").json()["experimentos"] == []  # no se lanzó nada
    assert cliente.post("/api/marcos/marco-sintetico/comparativa", json={"estrategias": []}).status_code == 422
    assert cliente.post("/api/marcos/nada/comparativa", json={"estrategias": [{"estrategia": "pesos_iguales"}]}).status_code == 404
    tfm = cliente.post("/api/marcos/marco-sintetico/comparativa", json={"estrategias": [{"estrategia": "tfm_bayes_stein"}]})
    assert tfm.status_code == 422 and "no se puede lanzar" in tfm.json()["detail"]


def test_comparar_avisa_cuando_no_es_ceteris_paribus(tmp_path):
    cliente = _cliente(tmp_path)
    base = {"estrategia": "pesos_iguales", "universo": TICKERS, "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}
    a = cliente.post("/api/experimentos", json=base).json()["id"]
    b = cliente.post("/api/experimentos", json={**base, "fecha_fin": "2021-06-30", "comision": 0.003}).json()["id"]
    for i in (a, b):
        _esperar(cliente, i)
    comparacion = cliente.get("/api/comparar", params={"ids": f"{a},{b}"}).json()
    assert comparacion["ceteris_paribus"]["ok"] is False
    assert {d["campo"] for d in comparacion["ceteris_paribus"]["diferencias"]} == {"fecha_fin", "comision"}
    # Un solo experimento: nada que comparar
    assert cliente.get("/api/comparar", params={"ids": a}).json()["ceteris_paribus"]["ok"] is True
