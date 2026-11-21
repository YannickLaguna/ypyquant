"""
test_api_comparar_series.py — Tests de `GET /api/comparar/series` (series temporales por experimento) con precios
sintéticos: drawdown, Sharpe y volatilidad móviles sobre el eje común, turnover por rebalanceo y «sin dato» cuando
el experimento no tiene pesos (como los importados del TFM).

Se relaciona con: app/api/rutas_experimentos.py, ypyquant.estrategias.series_comparacion y tests/test_api_marcos.py.
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.api.main import crear_app
from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.estrategias.precios import CargadorSintetico

TICKERS = ["AAA", "BBB", "CCC", "DDD"]
BASE = {"universo": TICKERS, "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}


@pytest.fixture(autouse=True)
def sin_claves(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)


@pytest.fixture()
def entorno(tmp_path):
    app = crear_app(CargadorSintetico(), tmp_path / "exp", ServicioAlpaca(), BotControl(tmp_path / "paper"), cargar_env=False)
    cliente = TestClient(app)
    ids = []
    for estrategia, parametros in (("pesos_iguales", {}), ("momentum", {"top_k": 2})):
        id_experimento = cliente.post("/api/experimentos", json={**BASE, "estrategia": estrategia, "parametros": parametros}).json()["id"]
        limite = time.time() + 90
        while time.time() < limite and cliente.get(f"/api/experimentos/{id_experimento}").json()["estado"] not in ("hecho", "error"):
            time.sleep(0.1)
        ids.append(id_experimento)
    return cliente, ids, tmp_path / "exp"


def _serie(cliente, ids, indicador):
    return cliente.get("/api/comparar/series", params={"ids": ",".join(ids), "indicador": indicador})


def test_drawdown_y_moviles_comparten_el_eje_comun(entorno):
    cliente, ids, _ = entorno
    comparacion = cliente.get("/api/comparar", params={"ids": ",".join(ids)}).json()
    drawdown = _serie(cliente, ids, "drawdown")
    assert drawdown.status_code == 200
    cuerpo = drawdown.json()
    assert cuerpo["indicador"] == "drawdown" and cuerpo["ids"] == ids and cuerpo["sin_dato"] == []
    for i in ids:
        assert cuerpo["series"][i]["fechas"] == comparacion["fechas"]
        assert max(cuerpo["series"][i]["valores"]) == 0.0 and min(cuerpo["series"][i]["valores"]) <= 0.0
    # Móviles a 12 meses: diaria = 252 observaciones; antes de completarse la ventana, None
    sharpe = _serie(cliente, ids, "sharpe_movil").json()["series"][ids[0]]["valores"]
    assert len(sharpe) == len(comparacion["fechas"]) and sharpe[:252] == [None] * 252 and sharpe[-1] is not None
    volatilidad = _serie(cliente, ids, "volatilidad_movil").json()["series"][ids[0]]["valores"]
    assert volatilidad[251] is None and volatilidad[252] is not None and volatilidad[252] > 0


def test_turnover_por_rebalanceo_y_sin_dato_si_faltan_pesos(entorno):
    cliente, ids, raiz = entorno
    cuerpo = _serie(cliente, ids, "turnover_rebalanceo").json()
    assert cuerpo["sin_dato"] == []
    momentum = cuerpo["series"][ids[1]]
    assert len(momentum["fechas"]) == len(momentum["valores"]) > 5 and all(0 <= v <= 1 for v in momentum["valores"])
    # Un experimento sin pesos (como un importado del TFM) queda «sin dato», no a cero
    (raiz / ids[0] / "pesos.parquet").unlink()
    cuerpo = _serie(cliente, ids, "turnover_rebalanceo").json()
    assert cuerpo["sin_dato"] == [ids[0]] and cuerpo["series"][ids[0]] == {"fechas": [], "valores": []}


def test_errores_de_la_ruta(entorno):
    cliente, ids, _ = entorno
    assert _serie(cliente, ids, "inventado").status_code == 422
    assert _serie(cliente, ids, "equity").status_code == 422  # la equity va por /api/comparar
    assert _serie(cliente, ["no-existe"], "drawdown").status_code == 404
    assert cliente.get("/api/comparar/series", params={"ids": ",".join(ids)}).status_code == 422  # falta el indicador
