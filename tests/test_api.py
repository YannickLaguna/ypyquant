"""
Tests de la API FastAPI (app/api) con TestClient: estrategias, experimentos, comparación y paper trading.

Usa precios sintéticos, un almacén temporal, un cliente Alpaca falso y un bot de mentira (subproceso
trivial): no necesita red, claves ni Alpaca real.

Se relaciona con: app/api/*, ypyquant.estrategias y tests/test_estrategias.py.
"""

import sys
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.main import crear_app
from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.estrategias.precios import CargadorSintetico

TICKERS = ["AAA", "BBB", "CCC", "DDD"]
CONFIG = {"estrategia": "pesos_iguales", "universo": TICKERS, "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}


@pytest.fixture(autouse=True)
def sin_claves(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)


def _cliente(tmp_path, alpaca=None, bot=None) -> TestClient:
    app = crear_app(CargadorSintetico(), tmp_path / "exp", alpaca or ServicioAlpaca(),
                    bot or BotControl(tmp_path / "paper"), cargar_env=False)
    return TestClient(app)


def _esperar(cliente: TestClient, id_experimento: str, timeout: float = 60.0) -> dict:
    limite = time.time() + timeout
    while time.time() < limite:
        detalle = cliente.get(f"/api/experimentos/{id_experimento}").json()
        if detalle["estado"] in ("hecho", "error"):
            return detalle
        time.sleep(0.1)
    raise AssertionError("El experimento no terminó a tiempo")


# ---- estrategias ----------------------------------------------------------------


def test_salud_y_host_no_local_rechazado(tmp_path):
    cliente = _cliente(tmp_path)
    assert cliente.get("/api/salud").json()["estado"] == "ok"
    assert cliente.get("/api/salud", headers={"host": "evil.example.com"}).status_code == 400


def test_listar_y_detallar_estrategias(tmp_path):
    cliente = _cliente(tmp_path)
    cuerpo = cliente.get("/api/estrategias").json()
    por_nombre = {e["nombre"]: e for e in cuerpo["estrategias"]}
    assert {"momentum", "bayes_stein", "inversa_volatilidad"} <= set(por_nombre)
    assert por_nombre["momentum"]["parametros_schema"]["properties"]["top_k"]["minimum"] == 1

    detalle = cliente.get("/api/estrategias/momentum").json()
    assert "class Momentum" in detalle["codigo_fuente"] and detalle["origen"] == "incluida"
    assert cliente.get("/api/estrategias/inversa_volatilidad").json()["origen"] == "usuario"
    assert cliente.get("/api/estrategias/nada").status_code == 404
    assert cliente.get("/api/universos").json()["universos"][0]["nombre"] == "sp500"
    assert cliente.post("/api/estrategias/recargar").status_code == 200
    assert cliente.get("/api/ruta-inexistente").status_code in (404, 405)


# ---- experimentos ----------------------------------------------------------------


def test_lanzar_leer_reutilizar_y_borrar_experimento(tmp_path):
    cliente = _cliente(tmp_path)
    respuesta = cliente.post("/api/experimentos", json=CONFIG)
    assert respuesta.status_code == 202
    id_experimento = respuesta.json()["id"]
    assert respuesta.json()["reutilizado"] is False

    detalle = _esperar(cliente, id_experimento)
    assert detalle["estado"] == "hecho", detalle["error"]
    assert detalle["metricas"]["sharpe"] is not None and detalle["equity"] and detalle["pesos"] and detalle["operaciones"]
    assert {"fecha", "equity", "benchmark", "drawdown"} <= set(detalle["equity"][0])
    assert detalle["config"]["parametros"] == {}  # solo se guarda la config normalizada (pesos_iguales no tiene)

    assert cliente.post("/api/experimentos", json=CONFIG).json() == {"id": id_experimento, "reutilizado": True}
    assert cliente.post("/api/experimentos?forzar=true", json=CONFIG).json()["reutilizado"] is False
    _esperar(cliente, id_experimento)

    lista = cliente.get("/api/experimentos", params={"estrategia": "pesos_iguales", "estado": "hecho"}).json()
    assert [e["id"] for e in lista["experimentos"]] == [id_experimento]
    assert cliente.get("/api/experimentos", params={"estrategia": "momentum"}).json()["experimentos"] == []

    assert cliente.delete(f"/api/experimentos/{id_experimento}").status_code == 204
    assert cliente.delete(f"/api/experimentos/{id_experimento}").status_code == 404


def test_borrar_varios_experimentos_omite_los_que_siguen_en_marcha(tmp_path):
    cliente = _cliente(tmp_path)
    ids = [cliente.post("/api/experimentos", json={**CONFIG, "parametros": {}, "comision": c}).json()["id"]
           for c in (0.001, 0.002)]
    for id_experimento in ids:
        assert _esperar(cliente, id_experimento)["estado"] == "hecho"
    # Un experimento «en curso» no se borra: el ejecutor seguiría escribiendo en su carpeta.
    en_marcha = cliente.post("/api/experimentos", json={**CONFIG, "comision": 0.003}).json()["id"]
    _esperar(cliente, en_marcha)
    cliente.app.state.ctx.almacen.marcar_estado(en_marcha, "en_curso")

    respuesta = cliente.post("/api/experimentos/borrar", json={"ids": [*ids, en_marcha, "no-existe"]})
    assert respuesta.status_code == 200
    assert respuesta.json() == {"borrados": ids, "en_marcha": [en_marcha], "no_encontrados": ["no-existe"]}
    restantes = [e["id"] for e in cliente.get("/api/experimentos").json()["experimentos"]]
    assert restantes == [en_marcha]
    assert cliente.post("/api/experimentos/borrar", json={"ids": []}).status_code == 422
    assert cliente.get(f"/api/experimentos/{id_experimento}").status_code == 404


def test_validacion_de_experimentos(tmp_path):
    cliente = _cliente(tmp_path)
    assert cliente.post("/api/experimentos", json={**CONFIG, "estrategia": "nada"}).status_code == 404
    assert cliente.post("/api/experimentos", json={**CONFIG, "estrategia": "momentum", "parametros": {"top_k": 0}}).status_code == 422
    assert cliente.post("/api/experimentos", json={**CONFIG, "fecha_inicio": "2022-01-01"}).status_code == 422
    assert cliente.post("/api/experimentos", json={"estrategia": "momentum"}).status_code == 422


def test_experimento_con_error_deja_traza(tmp_path):
    app = crear_app(lambda t, i, f: None, tmp_path / "exp", ServicioAlpaca(), BotControl(tmp_path / "p"), cargar_env=False)
    cliente = TestClient(app)
    id_experimento = cliente.post("/api/experimentos", json=CONFIG).json()["id"]
    detalle = _esperar(cliente, id_experimento)
    assert detalle["estado"] == "error" and "No hay precios" in detalle["error"] and "Traceback" in detalle["traza"]


def test_barrido_y_comparar(tmp_path):
    cliente = _cliente(tmp_path)
    cuerpo = {"config": {**CONFIG, "estrategia": "momentum"}, "rejilla": {"top_k": [1, 2, 3]}}
    respuesta = cliente.post("/api/experimentos/barrido", json=cuerpo)
    assert respuesta.status_code == 202 and respuesta.json()["total"] == 3
    ids = respuesta.json()["ids"]
    assert all(_esperar(cliente, i)["estado"] == "hecho" for i in ids)

    comparacion = cliente.get("/api/comparar", params={"ids": ",".join(ids)}).json()
    assert comparacion["ids"] == ids and len(comparacion["fechas"]) > 100
    assert all(abs(serie[0] - 100.0) < 1e-6 for serie in comparacion["series"].values())
    assert set(comparacion["metricas"]) == set(ids)
    assert cliente.get("/api/comparar", params={"ids": "no-existe"}).status_code == 404
    assert cliente.post("/api/experimentos/barrido", json={**cuerpo, "rejilla": {"top_k": [0]}}).status_code == 422


def test_experimentos_interrumpidos_se_marcan_al_arrancar(tmp_path):
    cliente = _cliente(tmp_path)
    id_experimento = cliente.post("/api/experimentos", json=CONFIG).json()["id"]
    _esperar(cliente, id_experimento)
    cliente.app.state.ctx.almacen.marcar_estado(id_experimento, "en_curso")
    nuevo = _cliente(tmp_path)  # reinicio sobre el mismo almacén
    assert nuevo.get(f"/api/experimentos/{id_experimento}").json()["estado"] == "error"


# ---- paper trading ----------------------------------------------------------------


def test_paper_sin_claves_responde_503_con_mensaje(tmp_path):
    cliente = _cliente(tmp_path)
    assert cliente.get("/api/paper/estado").json()["alpaca_configurado"] is False
    for ruta in ("cuenta", "posiciones", "ordenes", "historial"):
        respuesta = cliente.get(f"/api/paper/{ruta}")
        assert respuesta.status_code == 503 and "ALPACA_API_KEY" in respuesta.json()["detail"]
    arranque = cliente.post("/api/paper/bot/arrancar", json={"estrategia": "momentum", "tickers": ["AAPL"]})
    assert arranque.status_code == 503
    assert cliente.get("/api/paper/fills").status_code == 200


class _ClienteAlpacaFalso:
    """Imita lo mínimo de TradingClient que usa ServicioAlpaca."""

    def get_account(self):
        return SimpleNamespace(account_number="PA1", status="ACTIVE", equity="101000", cash="50000",
                               buying_power="200000", portfolio_value="101000", last_equity="100000")

    def get_all_positions(self):
        return [SimpleNamespace(symbol="AAPL", qty="10", side="long", avg_entry_price="150", current_price="160",
                                market_value="1600", unrealized_pl="100", unrealized_plpc="0.066")]

    def get_orders(self, _peticion):
        return [SimpleNamespace(id="abc", symbol="AAPL", side=SimpleNamespace(value="buy"), order_type="market",
                                qty="10", filled_qty="10", filled_avg_price="150", status=SimpleNamespace(value="filled"),
                                created_at=None, filled_at=None)]

    def get_portfolio_history(self, _peticion):
        return SimpleNamespace(timestamp=[1_700_000_000, 1_700_086_400], equity=[100000.0, 101000.0],
                               profit_loss=[0.0, 1000.0], base_value=100000.0)


def test_paper_con_cliente_alpaca_falso(tmp_path):
    cliente = _cliente(tmp_path, alpaca=ServicioAlpaca(fabrica_cliente=_ClienteAlpacaFalso))
    cuenta = cliente.get("/api/paper/cuenta").json()
    assert cuenta["equity"] == 101000.0 and cuenta["pnl_dia"] == 1000.0 and cuenta["pnl_dia_pct"] == pytest.approx(0.01)
    assert cliente.get("/api/paper/posiciones").json()["posiciones"][0]["ticker"] == "AAPL"
    assert cliente.get("/api/paper/ordenes").json()["ordenes"][0]["lado"] == "buy"
    historial = cliente.get("/api/paper/historial").json()
    assert historial["equity"] == [100000.0, 101000.0] and len(historial["fechas"]) == 2


def test_paper_error_de_alpaca_es_502(tmp_path):
    def fabrica():
        raise ConnectionError("sin red")

    cliente = _cliente(tmp_path, alpaca=ServicioAlpaca(fabrica_cliente=fabrica))
    respuesta = cliente.get("/api/paper/cuenta")
    assert respuesta.status_code == 502 and "sin red" in respuesta.json()["detail"]


def test_bot_arrancar_log_y_parar(tmp_path):
    def comando(estrategia, parametros, tickers, intervalo):
        codigo = "import sys, time; print('hola', *sys.argv[1:], flush=True); time.sleep(60)"
        return [sys.executable, "-c", codigo, estrategia, str(tickers), str(parametros)]

    bot = BotControl(tmp_path / "paper", construir_comando=comando)
    cliente = _cliente(tmp_path, alpaca=ServicioAlpaca(fabrica_cliente=_ClienteAlpacaFalso), bot=bot)
    try:
        assert cliente.post("/api/paper/bot/arrancar", json={"estrategia": "nada", "tickers": ["A"]}).status_code == 404
        assert cliente.post("/api/paper/bot/arrancar",
                            json={"estrategia": "momentum", "parametros": {"top_k": 0}, "tickers": ["A"]}).status_code == 422
        respuesta = cliente.post("/api/paper/bot/arrancar",
                                 json={"estrategia": "momentum", "parametros": {"top_k": 2}, "tickers": ["aapl", "msft"]})
        assert respuesta.status_code == 200 and respuesta.json()["en_marcha"] is True
        assert cliente.post("/api/paper/bot/arrancar", json={"estrategia": "momentum", "tickers": ["A"]}).status_code == 409

        lineas = []
        for _ in range(100):
            lineas = cliente.get("/api/paper/bot/log").json()["lineas"]
            if lineas:
                break
            time.sleep(0.1)
        assert "hola momentum ['AAPL', 'MSFT']" in lineas[0]
        assert cliente.get("/api/paper/bot/log", params={"desde": 1}).json()["lineas"] == []
        assert cliente.get("/api/paper/estado").json()["bot"]["estrategia"] == "momentum"
    finally:
        parado = cliente.post("/api/paper/bot/parar").json()
    assert parado["en_marcha"] is False
    assert cliente.get("/api/paper/bot/estado").json()["en_marcha"] is False


def test_bot_recibe_dias_de_historia_y_frecuencia_de_la_peticion(tmp_path):
    """`dias_historico` y `frecuencia_rebalanceo` de la petición llegan al comando del bot (solo si se indican)."""
    recibidos = []

    def comando(estrategia, parametros, tickers, intervalo, **extras):
        recibidos.append(extras)
        return [sys.executable, "-c", "import time; time.sleep(60)"]

    bot = BotControl(tmp_path / "paper", construir_comando=comando)
    cliente = _cliente(tmp_path, alpaca=ServicioAlpaca(fabrica_cliente=_ClienteAlpacaFalso), bot=bot)
    try:
        # Valores fuera de rango o frecuencia desconocida: 422 y no se lanza nada
        for malo in ({"dias_historico": 5}, {"dias_historico": 99_999}, {"frecuencia_rebalanceo": "anual"}):
            assert cliente.post("/api/paper/bot/arrancar", json={"estrategia": "momentum", "tickers": ["A"], **malo}).status_code == 422
        assert recibidos == []
        respuesta = cliente.post("/api/paper/bot/arrancar", json={
            "estrategia": "momentum", "tickers": ["A"], "dias_historico": 600, "frecuencia_rebalanceo": "weekly"})
        assert respuesta.status_code == 200
        assert recibidos == [{"dias_historico": 600, "frecuencia": "weekly"}]
        assert respuesta.json()["dias_historico"] == 600 and respuesta.json()["frecuencia_rebalanceo"] == "weekly"
    finally:
        cliente.post("/api/paper/bot/parar")


def test_comando_por_defecto_del_bot_incluye_dias_y_frecuencia_solo_si_se_piden():
    base = BotControl._comando_por_defecto("momentum", {"top_k": 3}, ["AAPL"], 60)
    assert "--dias-historico" not in base and "--frecuencia" not in base
    completo = BotControl._comando_por_defecto("momentum", {}, ["AAPL"], 60, dias_historico=600, frecuencia="monthly")
    assert completo[completo.index("--dias-historico") + 1] == "600" and completo[completo.index("--frecuencia") + 1] == "monthly"
