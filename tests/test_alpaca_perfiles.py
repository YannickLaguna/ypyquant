"""
test_alpaca_perfiles.py — Tests de Alpaca multi-perfil (fase 5): perfiles, redacción, ServicioAlpaca, GestorBots, vínculo y rutas.

Todo con clientes falsos y procesos de mentira: no se llama a Alpaca ni se leen claves reales (se fijan variables de
entorno de prueba con `monkeypatch`). Cubre las tareas 5.1 (descubrir perfiles sin exponer secretos), 5.2
(`ServicioAlpaca(perfil)`), 5.3 (`GestorBots`: dos perfiles a la vez, 409, rehidratación), 5.5 (fills antiguos) y 5.6
(rutas con `?perfil=` y de `/alpaca/`).

Se relaciona con: ypyquant.perfiles_alpaca, app/api/paper.py, app/api/gestor_bots.py, app/api/rutas_alpaca.py y
tests/test_api.py (los tests de siempre con el perfil por defecto no cambian).
"""

import json
import logging
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.gestor_bots import GestorBots, LimiteBotsActivos, PerfilOcupado
from app.api.main import crear_app
from app.api.paper import BotControl, ServicioAlpaca, SinClavesAlpaca, leer_fills
from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.perfiles_alpaca.almacen import AlmacenPerfiles
from ypyquant.perfiles_alpaca.perfiles import claves_de, descubrir_perfiles, nombre_valido, variables_de
from ypyquant.perfiles_alpaca.procesos import marca_de_proceso, proceso_vivo, terminar
from ypyquant.perfiles_alpaca.redaccion import FiltroRedaccion, redactar, secretos_del_entorno
from ypyquant.perfiles_alpaca.vinculo import UniversoNoSoportado, receta_de_origen, resolver_tickers, vincular

CLAVE_A, SECRETO_A = "PKAAAAAAAAAAAAAAAAAA1234", "secretoDeLaCuentaA0123456789abcdefghijkl"
CLAVE_B, SECRETO_B = "PKBBBBBBBBBBBBBBBBBB5678", "secretoDeLaCuentaB0123456789abcdefghijkl"
ENTORNO = {"ALPACA_API_KEY": "PKDEFECTODEFECTODEFECTO0001", "ALPACA_SECRET_KEY": "secretoDefecto0123456789abcdefghijklmn",
           "ALPACA_A_API_KEY": CLAVE_A, "ALPACA_A_SECRET_KEY": SECRETO_A,
           "ALPACA_B_API_KEY": CLAVE_B, "ALPACA_B_SECRET_KEY": SECRETO_B,
           "ALPACA_INCOMPLETO_API_KEY": "PKINCOMPLETO0000000000"}
VARIABLES_ALPACA = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PERFILES")


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    """Quita cualquier variable ALPACA_* real del entorno: los tests fijan las suyas."""
    import os

    for variable in [v for v in os.environ if v.startswith("ALPACA_")]:
        monkeypatch.delenv(variable, raising=False)


def _poner(monkeypatch, entorno):
    for k, v in entorno.items():
        monkeypatch.setenv(k, v)


# ---- 5.1 perfiles --------------------------------------------------------------------------------


def test_descubre_defecto_y_perfiles_con_nombre_ignorando_incompletos_como_no_configurados():
    perfiles = {p.nombre: p for p in descubrir_perfiles(ENTORNO)}
    assert list(perfiles) == ["DEFECTO", "A", "B", "INCOMPLETO"]  # DEFECTO primero, luego alfabético
    assert perfiles["A"].configurado and perfiles["A"].ultimos4 == "1234" and perfiles["B"].ultimos4 == "5678"
    assert not perfiles["INCOMPLETO"].configurado and perfiles["INCOMPLETO"].faltan == ("ALPACA_INCOMPLETO_SECRET_KEY",)
    assert claves_de("INCOMPLETO", ENTORNO) is None  # las variables incompletas nunca se usan
    assert claves_de("A", ENTORNO) == (CLAVE_A, SECRETO_A)


def test_perfiles_declarados_sin_claves_y_vacio_total():
    declarados = {p.nombre: p for p in descubrir_perfiles({"ALPACA_PERFILES": "alfa, beta;gamma"})}
    assert set(declarados) == {"ALFA", "BETA", "GAMMA"} and not any(p.configurado for p in declarados.values())
    assert declarados["ALFA"].faltan == ("ALPACA_ALFA_API_KEY", "ALPACA_ALFA_SECRET_KEY")
    vacio = descubrir_perfiles({})
    assert [p.nombre for p in vacio] == ["DEFECTO"] and not vacio[0].configurado  # la UI explica qué definir


def test_los_perfiles_no_exponen_secretos_ni_la_api_key_completa():
    texto = json.dumps([p.a_dict() for p in descubrir_perfiles(ENTORNO)])
    for secreto in (*ENTORNO.values(),):
        assert secreto not in texto
    assert "1234" in texto  # solo los últimos 4 de la API key


def test_nombres_validos_y_variables():
    assert nombre_valido("A") and nombre_valido("CUENTA_2") and not nombre_valido("a") and not nombre_valido("../X")
    assert not nombre_valido("_X") and not nombre_valido("")
    assert variables_de("DEFECTO") == ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
    assert variables_de("A") == ("ALPACA_A_API_KEY", "ALPACA_A_SECRET_KEY")


# ---- redacción -----------------------------------------------------------------------------------


def test_redactar_oculta_secretos_del_entorno_y_claves_con_forma_de_api_key(monkeypatch):
    _poner(monkeypatch, ENTORNO)
    assert SECRETO_A in secretos_del_entorno()
    texto = f"fallo con {SECRETO_A} y la clave {CLAVE_B}, y otra PKZZZZZZZZZZZZZZZZZZZ99"
    limpio = redactar(texto)
    assert SECRETO_A not in limpio and CLAVE_B not in limpio and "PKZZZ" not in limpio and "***" in limpio
    assert redactar("sin nada secreto") == "sin nada secreto"


def test_filtro_de_logging_redacta_mensaje_argumentos_y_traza(monkeypatch, caplog):
    _poner(monkeypatch, ENTORNO)
    logger = logging.getLogger("prueba_redaccion")
    logger.addFilter(FiltroRedaccion())
    with caplog.at_level(logging.INFO, logger="prueba_redaccion"):
        logger.info("clave %s", SECRETO_B)
        try:
            raise RuntimeError(f"error con {SECRETO_A}")
        except RuntimeError:
            logger.exception("falló")
    salida = caplog.text
    assert SECRETO_A not in salida and SECRETO_B not in salida and "***" in salida


# ---- 5.2 ServicioAlpaca(perfil) ------------------------------------------------------------------


class _Cuenta:
    def __init__(self, equity):
        self.equity = equity

    def get_account(self):
        return SimpleNamespace(account_number="PA", status="ACTIVE", equity=str(self.equity), cash="1", buying_power="1",
                               portfolio_value=str(self.equity), last_equity=str(self.equity))


def test_cada_perfil_usa_sus_propias_claves(monkeypatch):
    _poner(monkeypatch, ENTORNO)
    import alpaca.trading.client as modulo

    creados = []

    class ClienteFalso:
        def __init__(self, api_key, secret_key, paper):
            creados.append((api_key, secret_key, paper))

    monkeypatch.setattr(modulo, "TradingClient", ClienteFalso)
    ServicioAlpaca(perfil="A").cliente()
    ServicioAlpaca(perfil="B").cliente()
    ServicioAlpaca().cliente()
    assert creados[0] == (CLAVE_A, SECRETO_A, True) and creados[1] == (CLAVE_B, SECRETO_B, True)
    assert creados[2][0] == ENTORNO["ALPACA_API_KEY"] and all(c[2] for c in creados)  # siempre paper


def test_perfil_sin_claves_lanza_error_con_su_nombre_y_variables(monkeypatch):
    servicio = ServicioAlpaca(perfil="C")
    assert not servicio.disponible
    with pytest.raises(SinClavesAlpaca, match=r"C.*ALPACA_C_API_KEY.*ALPACA_C_SECRET_KEY"):
        servicio.cuenta()


def test_comprobar_conexion_resume_sin_claves_y_redacta_errores(monkeypatch):
    assert ServicioAlpaca(perfil="C").comprobar_conexion() == {"configurada": False, "conectada": False, "equity": None, "error": None}
    ok = ServicioAlpaca(fabrica_cliente=lambda: _Cuenta(5000), perfil="A").comprobar_conexion()
    assert ok == {"configurada": True, "conectada": True, "equity": 5000.0, "error": None}

    def fabrica():
        raise RuntimeError(f"401 con la clave {CLAVE_A}")

    caido = ServicioAlpaca(fabrica_cliente=fabrica, perfil="A").comprobar_conexion()
    assert caido["conectada"] is False and CLAVE_A not in caido["error"]


# ---- 5.5 fills antiguos y por perfil -------------------------------------------------------------


def test_leer_fills_por_perfil_y_retrocompatibilidad(tmp_path):
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "fills.jsonl").write_text(
        json.dumps({"ticker": "X", "lado": "compra", "cantidad": 1, "precio": 2, "fecha": "2026-01-01", "id_orden": "1"}) + "\n"
        + "{corrupta\n"
        + json.dumps({"ticker": "Y", "perfil": "A", "experimento_id": "e1", "precio_referencia": 3, "tipo": "rebalanceo",
                      "id_orden": "2"}) + "\n", encoding="utf-8")
    filas = leer_fills(perfil="A", carpeta=tmp_path)
    assert [f["ticker"] for f in filas] == ["Y", "X"] and "perfil" not in filas[1]  # fila antigua, sin campos nuevos
    assert leer_fills(perfil="B", carpeta=tmp_path) == []
    (tmp_path / "fills.jsonl").write_text(json.dumps({"ticker": "Z", "id_orden": "9"}) + "\n", encoding="utf-8")
    assert leer_fills(perfil="DEFECTO", carpeta=tmp_path)[0]["ticker"] == "Z"  # DEFECTO cae al fichero antiguo


# ---- 5.3 GestorBots ------------------------------------------------------------------------------


def _dormir(*_):
    return [sys.executable, "-c", "import time; print('hola', flush=True); time.sleep(60)"]


@pytest.fixture
def gestor(tmp_path):
    g = GestorBots(AlmacenPerfiles(tmp_path / "pt"), construir_comando=_dormir, limite_activos=2)
    yield g
    g.parar_todos(espera_s=3)


def test_dos_perfiles_arrancan_a_la_vez_y_el_ocupado_devuelve_409(gestor):
    a = gestor.arrancar("A", experimento_id="e1")
    b = gestor.arrancar("B", experimento_id="e2", modo="diario", intervalo_s=30)
    assert a["en_marcha"] and b["en_marcha"] and a["pid"] != b["pid"] and b["modo"] == "diario"
    assert gestor.activos() == ["A", "B"]
    with pytest.raises(PerfilOcupado):
        gestor.arrancar("A", experimento_id="e1")
    with pytest.raises(LimiteBotsActivos):  # límite blando (aquí 2)
        gestor.arrancar("C", experimento_id="e3")
    assert not gestor.parar("A")["en_marcha"] and gestor.estado("B")["en_marcha"]
    assert gestor.almacen.leer_bot("A") is None  # al parar se limpia bot.json


def test_el_bot_deja_bot_json_con_pid_y_marca_y_el_log_se_redacta(gestor, monkeypatch):
    monkeypatch.setenv("ALPACA_A_SECRET_KEY", SECRETO_A)
    gestor.arrancar("A", experimento_id="e1")
    meta = gestor.almacen.leer_bot("A")
    assert meta["pid"] and meta["experimento_id"] == "e1" and meta["modo"] == "continuo"
    gestor.almacen.ruta_log("A").write_text(f"hola\nclave {SECRETO_A}\n", encoding="utf-8")
    log = gestor.log("A")
    assert log["lineas"] == ["hola", "clave ***"] and log["siguiente"] == 2
    assert gestor.log("A", desde=2)["lineas"] == []


def test_rehidratacion_adopta_pid_vivo_y_descarta_el_muerto(tmp_path):
    almacen = AlmacenPerfiles(tmp_path / "pt")
    vivo = subprocess.Popen(_dormir())
    muerto = subprocess.Popen([sys.executable, "-c", "pass"])
    muerto.wait()
    try:
        almacen.guardar_bot("A", {"pid": vivo.pid, "marca_proceso": marca_de_proceso(vivo.pid), "modo": "continuo",
                                  "intervalo_s": 60, "experimento_id": "e1"})
        almacen.guardar_bot("B", {"pid": muerto.pid, "marca_proceso": None, "modo": "continuo"})
        almacen.guardar_bot("C", {"pid": vivo.pid, "marca_proceso": 1.0, "modo": "continuo"})  # pid reutilizado: otra hora de creación
        gestor = GestorBots(almacen, construir_comando=_dormir)
        adoptados = gestor.rehidratar()
        if marca_de_proceso(vivo.pid) is None:  # sin psutil no se puede distinguir un pid reutilizado
            assert adoptados == ["A", "C"]
        else:
            assert adoptados == ["A"] and almacen.leer_bot("C") is None
        assert gestor.estado("A")["en_marcha"] and gestor.estado("A")["adoptado"] and gestor.estado("A")["experimento_id"] == "e1"
        assert almacen.leer_bot("B") is None and not gestor.estado("B")["en_marcha"]
        assert not gestor.parar("A")["en_marcha"]  # se para por pid, sin Popen
        vivo.wait(timeout=10)
    finally:
        if vivo.poll() is None:
            vivo.kill()


def test_proceso_vivo_y_terminar():
    p = subprocess.Popen(_dormir())
    try:
        assert proceso_vivo(p.pid, marca_de_proceso(p.pid))
        assert not proceso_vivo(None) and not proceso_vivo(0)
        assert terminar(p.pid, espera_s=5)
        p.wait(timeout=10)
        assert not proceso_vivo(p.pid)
    finally:
        if p.poll() is None:
            p.kill()


# ---- almacén y vínculo ---------------------------------------------------------------------------

CONFIG = {"estrategia": "momentum", "parametros": {"top_k": 2}, "universo": ["aaa", "BBB", "CCC", "DDD"],
          "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}


def test_instantaneas_una_fila_por_dia_con_upsert(tmp_path):
    almacen = AlmacenPerfiles(tmp_path)
    assert almacen.leer_instantaneas("A").empty
    almacen.guardar_instantanea("A", "2026-11-23", 100_000.0, 50_000.0, {"AAPL": {"cantidad": 10, "precio": 150.0}})
    almacen.guardar_instantanea("A", "2026-11-22", 99_000.0, 99_000.0, {})
    almacen.guardar_instantanea("A", "2026-11-23", 101_000.0, 40_000.0, {"AAPL": {"cantidad": 12, "precio": 160.0}})  # reemplaza
    tabla = almacen.leer_instantaneas("A")
    assert list(tabla["fecha"]) == ["2026-11-22", "2026-11-23"] and list(tabla["equity"]) == [99_000.0, 101_000.0]
    assert json.loads(tabla["posiciones"].iloc[1]) == {"AAPL": {"cantidad": 12, "precio": 160.0}}


def test_vincular_congela_receta_es_idempotente_y_guarda_el_anterior(tmp_path):
    almacen = AlmacenPerfiles(tmp_path)
    receta = ConfiguracionExperimento(**CONFIG).receta().normalizada()
    vinculo, cambio = vincular(almacen, "A", receta, "experimento", "momentum-1", 100_000.0)
    assert cambio and vinculo["tickers"] == ["AAA", "BBB", "CCC", "DDD"] and vinculo["frecuencia_rebalanceo"] == "monthly"
    assert vinculo["receta"]["parametros"]["top_k"] == 2 and vinculo["equity_inicial"] == 100_000.0
    igual, cambio = vincular(almacen, "A", receta, "experimento", "momentum-1", 5.0)
    assert not cambio and igual["fecha_alta"] == vinculo["fecha_alta"] and igual["equity_inicial"] == 100_000.0
    almacen.guardar_estado_estrategia("A", {"fecha": "2026-11-20"})
    otra = ConfiguracionExperimento(**{**CONFIG, "parametros": {"top_k": 3}}).receta().normalizada()
    nuevo, cambio = vincular(almacen, "A", otra, "experimento", "momentum-2", 90_000.0)
    assert cambio and nuevo["origen"]["id"] == "momentum-2" and nuevo["vinculos_anteriores"][0]["origen"]["id"] == "momentum-1"
    assert almacen.leer_estado_estrategia("A") is None  # la memoria era de la estrategia anterior


def test_universo_dinamico_no_se_puede_operar_en_vivo():
    receta = ConfiguracionExperimento(**{**CONFIG, "universo": "sp500_pit"}).receta().normalizada()
    with pytest.raises(UniversoNoSoportado):
        resolver_tickers(receta)


def test_receta_de_origen_experimento_inexistente(tmp_path):
    from ypyquant.estrategias.almacen import AlmacenExperimentos

    almacen = AlmacenExperimentos(tmp_path / "exp")
    with pytest.raises(KeyError):
        receta_de_origen("experimento", "no-existe", almacen)
    id_experimento = almacen.registrar(ConfiguracionExperimento(**CONFIG))
    assert receta_de_origen("experimento", id_experimento, almacen).parametros["top_k"] == 2


# ---- 5.6 rutas -----------------------------------------------------------------------------------


class _AlpacaFalso:
    def __init__(self, equity):
        self.equity = equity

    def get_account(self):
        return SimpleNamespace(account_number="PA", status="ACTIVE", equity=str(self.equity), cash="10", buying_power="20",
                               portfolio_value=str(self.equity), last_equity=str(self.equity))

    def get_all_positions(self):
        return [SimpleNamespace(symbol=f"T{int(self.equity)}", qty="1", side="long", avg_entry_price="1", current_price="1",
                                market_value="1", unrealized_pl="0", unrealized_plpc="0")]


def _api(tmp_path, gestor=None):
    app = crear_app(FuenteSintetica(), tmp_path / "exp", ServicioAlpaca(), BotControl(tmp_path / "paper"), cargar_env=False)
    ctx = app.state.ctx
    if gestor:
        ctx.gestor_bots = gestor(ctx.almacen_perfiles)
    ctx.alpacas["A"] = ServicioAlpaca(fabrica_cliente=lambda: _AlpacaFalso(1000), perfil="A")
    ctx.alpacas["B"] = ServicioAlpaca(fabrica_cliente=lambda: _AlpacaFalso(2000), perfil="B")
    return TestClient(app), ctx


def test_rutas_con_perfil_usan_la_cuenta_de_cada_perfil(tmp_path):
    cliente, _ = _api(tmp_path)
    assert cliente.get("/api/paper/cuenta", params={"perfil": "A"}).json()["equity"] == 1000.0
    assert cliente.get("/api/paper/cuenta", params={"perfil": "B"}).json()["equity"] == 2000.0
    assert cliente.get("/api/paper/posiciones", params={"perfil": "B"}).json()["posiciones"][0]["ticker"] == "T2000"
    respuesta = cliente.get("/api/paper/cuenta", params={"perfil": "C"})  # sin claves: 503 con el nombre del perfil
    assert respuesta.status_code == 503 and "C" in respuesta.json()["detail"] and "ALPACA_C_API_KEY" in respuesta.json()["detail"]
    assert cliente.get("/api/paper/cuenta", params={"perfil": "../x"}).status_code == 422
    assert cliente.get("/api/paper/estado", params={"perfil": "A"}).json()["alpaca_configurado"] is True
    assert cliente.get("/api/paper/estado").json()["alpaca_configurado"] is False  # DEFECTO sin claves


def test_listado_de_perfiles_sin_secretos(tmp_path, monkeypatch):
    _poner(monkeypatch, {"ALPACA_PERFILES": "A,B,C", "ALPACA_C_API_KEY": "PKCCCCCCCCCCCCCCCCCC9999"})
    cliente, _ = _api(tmp_path)
    respuesta = cliente.get("/api/paper/alpaca/perfiles")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    por_nombre = {p["nombre"]: p for p in cuerpo["perfiles"]}
    assert list(por_nombre) == ["A", "B", "C"]  # DEFECTO no aparece: no está configurado ni declarado
    assert por_nombre["A"]["configurado"] and por_nombre["A"]["conexion"] == {"configurada": True, "conectada": True, "equity": 1000.0, "error": None}
    assert not por_nombre["C"]["configurado"] and por_nombre["C"]["faltan"] == ["ALPACA_C_SECRET_KEY"]
    assert por_nombre["C"]["variables"] == ["ALPACA_C_API_KEY", "ALPACA_C_SECRET_KEY"] and por_nombre["C"]["ultimos4"] is None
    assert "PKCCCC" not in respuesta.text and cuerpo["limite_activos"] == 3 and cuerpo["activos"] == []
    sin_conexion = cliente.get("/api/paper/alpaca/perfiles", params={"conexion": "false"}).json()
    assert sin_conexion["perfiles"][0]["conexion"]["conectada"] is None


def _registrar(ctx, **cambios):
    return ctx.almacen.registrar(ConfiguracionExperimento(**{**CONFIG, **cambios}))


def test_arrancar_bot_de_perfil_valida_origen_y_aplica_409(tmp_path):
    cliente, ctx = _api(tmp_path, gestor=lambda almacen: GestorBots(almacen, construir_comando=_dormir, limite_activos=3))
    try:
        url = "/api/paper/alpaca/bots/A/arrancar"
        assert cliente.post(url, json={}).status_code == 422  # sin experimento
        assert cliente.post(url, json={"experimento_id": "x", "configuracion_id": "y"}).status_code == 422
        assert cliente.post(url, json={"experimento_id": "no-existe"}).status_code == 404
        assert cliente.post("/api/paper/alpaca/bots/C/arrancar", json={"experimento_id": "x"}).status_code == 503  # sin claves
        pit = _registrar(ctx, universo="sp500_pit")
        assert cliente.post(url, json={"experimento_id": pit}).status_code == 422  # universo dinámico
        bueno = _registrar(ctx)
        respuesta = cliente.post(url, json={"experimento_id": bueno, "modo": "diario", "intervalo_s": 30})
        assert respuesta.status_code == 200 and respuesta.json()["en_marcha"] and respuesta.json()["modo"] == "diario"
        assert cliente.post(url, json={"experimento_id": bueno}).status_code == 409  # perfil ocupado
        assert cliente.post("/api/paper/alpaca/bots/B/arrancar", json={"experimento_id": bueno}).status_code == 200
        perfiles = cliente.get("/api/paper/alpaca/perfiles", params={"conexion": "false"}).json()
        assert perfiles["activos"] == ["A", "B"]
        estado = cliente.get("/api/paper/alpaca/bots/A").json()
        assert estado["en_marcha"] and estado["experimento_id"] == bueno
        assert cliente.get("/api/paper/alpaca/bots/A/log").json()["lineas"] in ([], ["hola"])
        assert not cliente.post("/api/paper/alpaca/bots/A/parar").json()["en_marcha"]
    finally:
        ctx.gestor_bots.parar_todos(espera_s=3)


def test_limite_blando_de_bots_devuelve_409(tmp_path):
    cliente, ctx = _api(tmp_path, gestor=lambda almacen: GestorBots(almacen, construir_comando=_dormir, limite_activos=1))
    try:
        bueno = _registrar(ctx)
        assert cliente.post("/api/paper/alpaca/bots/A/arrancar", json={"experimento_id": bueno}).status_code == 200
        respuesta = cliente.post("/api/paper/alpaca/bots/B/arrancar", json={"experimento_id": bueno})
        assert respuesta.status_code == 409 and "límite" in respuesta.json()["detail"]
    finally:
        ctx.gestor_bots.parar_todos(espera_s=3)


def test_instantaneas_por_ruta(tmp_path):
    cliente, ctx = _api(tmp_path)
    assert cliente.get("/api/paper/alpaca/A/instantaneas").json() == {"instantaneas": []}
    ctx.almacen_perfiles.guardar_instantanea("A", "2026-11-22", 1000.0, 10.0, {})
    filas = cliente.get("/api/paper/alpaca/A/instantaneas").json()["instantaneas"]
    assert filas[0]["fecha"] == "2026-11-22" and filas[0]["equity"] == 1000.0


def test_receta_con_apalancamiento_se_rechaza_en_vinculo_y_en_la_ruta(tmp_path):
    from ypyquant.perfiles_alpaca.vinculo import RecetaNoOperable

    receta = ConfiguracionExperimento(**{**CONFIG, "parametros": {"top_k": 2}}).receta()
    apalancada = receta.model_copy(update={"parametros": {**receta.parametros, "apalancamiento_maximo": 1.5}})
    with pytest.raises(RecetaNoOperable, match="cuenta de margen"):
        vincular(AlmacenPerfiles(tmp_path), "A", apalancada, "experimento", "x", None)
    assert AlmacenPerfiles(tmp_path).leer_vinculo("A") is None
    vincular(AlmacenPerfiles(tmp_path), "A", receta.model_copy(update={"parametros": {**receta.parametros, "apalancamiento_maximo": 1}}),
             "experimento", "x", None)
