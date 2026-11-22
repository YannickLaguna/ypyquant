"""
test_bot_experimento.py — Tests del bot de Alpaca por perfil y experimento (fase 5), con cliente Alpaca falso y reloj inyectado.

Responsabilidad: comprobar sin red ni claves que el bot (pipelines/paper_trading) respeta la frecuencia de rebalanceo (un
día sin rebalanceo no envía; un día de rebalanceo envía una vez y no repite), conserva la memoria de la estrategia tras
reiniciar (`estado_estrategia.json`), enriquece los fills (perfil, experimento, tipo, precio de referencia), guarda la
instantánea diaria de la cuenta, vincula el perfil a un experimento congelando su receta y no deja claves en los logs.
Cubre las tareas 5.4 y 5.5 del plan.

Se relaciona con: pipelines/paper_trading (__main__, alpaca_vivo, config, estado_bot), ypyquant.perfiles_alpaca,
ypyquant.execution.nautilus.registro_fills y tests/test_paper_vivo.py (el doble de Alpaca se reutiliza de allí).
"""

import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipelines.paper_trading import alpaca_vivo
from pipelines.paper_trading.__main__ import crear_parser, preparar_perfil
from pipelines.paper_trading.config import ConfiguracionAlpaca, ConfiguracionEstrategia, ConfiguracionServicio
from pipelines.paper_trading.estado_bot import EstadoBot, restaurar_estado_estrategia, serializar_estado_estrategia
from tests.test_paper_vivo import ClienteAlpacaFalso, _precios
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.perfiles_alpaca.almacen import AlmacenPerfiles
from ypyquant.perfiles_alpaca.redaccion import FiltroRedaccion
from ypyquant.raiz_datos import RaizDatos

pytest.importorskip("alpaca")  # extra `trading`: se construyen MarketOrderRequest reales


def _reloj(texto):
    """Reloj fijo: `ahora()` devuelve las 15:00 UTC de la fecha dada."""
    momento = datetime.fromisoformat(texto).replace(hour=15, tzinfo=UTC)
    return lambda: momento


@pytest.fixture(autouse=True)
def sin_claves(monkeypatch):
    import os

    for variable in [v for v in os.environ if v.startswith("ALPACA_")]:
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Raíz de datos temporal, cliente Alpaca falso y descarga de precios falsa (la fija cada test en `entorno.precios`)."""
    cliente = ClienteAlpacaFalso()
    monkeypatch.setattr(alpaca_vivo, "crear_clientes", lambda servicio: (cliente, None))
    monkeypatch.setattr(alpaca_vivo, "ESPERA_FILLS_S", 0.0)  # no esperar a que se ejecuten las órdenes de mentira
    entorno = SimpleNamespace(raiz=RaizDatos(base=tmp_path), cliente=cliente, precios=lambda: _precios())
    monkeypatch.setattr(alpaca_vivo, "descargar_precios", lambda tickers, dias: entorno.precios())
    return entorno


def _servicio(entorno, perfil="A", experimento="exp-1"):
    servicio = ConfiguracionServicio.desde_entorno(con_alpaca=False, perfil=perfil, raiz_datos=entorno.raiz)
    servicio.alpaca = ConfiguracionAlpaca("k", "s")
    servicio.segundos_entre_iteraciones = 0
    servicio.etiquetas_fills["experimento_id"] = experimento
    return servicio


def _estrategia(nombre="pesos_iguales", parametros=None, frecuencia="weekly", tickers=("AAA", "BBB")):
    return ConfiguracionEstrategia(nombre=nombre, tickers=list(tickers), estrategia_registro=nombre,
                                   parametros_registro=parametros or {}, frecuencia_rebalanceo_registro=frecuencia,
                                   huella_receta="h1", origen={"tipo": "experimento", "id": "exp-1"})


def _pasada(entorno, servicio, estrategia, fecha, iteraciones=1, **opciones):
    alpaca_vivo.ejecutar_vivo(servicio, estrategia, alpaca_vivo.ControlParada(), max_iteraciones=iteraciones,
                              ahora=_reloj(fecha), **opciones)


# ---- 5.4 frecuencia ------------------------------------------------------------------------------


def test_respeta_la_frecuencia_un_martes_no_envia_y_el_dia_de_rebalanceo_envia_una_vez(entorno):
    servicio, estrategia, cliente = _servicio(entorno), _estrategia(frecuencia="weekly"), entorno.cliente
    _pasada(entorno, servicio, estrategia, "2026-11-16", iteraciones=2)  # lunes: rebalancea en la 1.ª y NO repite en la 2.ª
    assert len(cliente.enviadas) == 2
    cliente.ejecutar("orden-1", 100.0)
    cliente.ejecutar("orden-2", 100.0)
    cliente.posiciones = {"AAA": 500, "BBB": 500}
    _pasada(entorno, servicio, estrategia, "2026-11-17")  # martes: misma semana, no toca
    assert len(cliente.enviadas) == 2
    _pasada(entorno, servicio, estrategia, "2026-11-23")  # lunes siguiente: semana nueva, toca (sin cambios de pesos: sin órdenes)
    estado = EstadoBot.cargar(servicio.ruta_estado, alpaca_vivo._firma_estrategia(estrategia))
    assert str(estado.ultimo_rebalanceo) == "2026-11-23"


def test_dia_sin_rebalanceo_no_descarga_precios_ni_envia(entorno):
    servicio, estrategia, cliente = _servicio(entorno), _estrategia(frecuencia="monthly"), entorno.cliente
    _pasada(entorno, servicio, estrategia, "2026-11-03")
    cliente.ejecutar("orden-1", 100.0)
    cliente.ejecutar("orden-2", 100.0)
    cliente.posiciones = {"AAA": 500, "BBB": 500}
    entorno.precios = lambda: pytest.fail("no debe descargar precios un día sin rebalanceo")
    _pasada(entorno, servicio, estrategia, "2026-11-04")  # martes del mismo mes
    assert len(cliente.enviadas) == 2


def test_con_mercado_cerrado_no_envia_salvo_que_se_permita(entorno):
    entorno.cliente.abierto = False
    servicio, estrategia = _servicio(entorno), _estrategia(frecuencia="daily")
    _pasada(entorno, servicio, estrategia, "2026-11-16")
    assert entorno.cliente.enviadas == []
    _pasada(entorno, servicio, estrategia, "2026-11-16", permitir_cerrado=True)  # --enviar-fuera-de-horario
    assert len(entorno.cliente.enviadas) == 2


# ---- 5.4 memoria de la estrategia tras reiniciar -------------------------------------------------


def _alcista(n=40, fin="2026-11-19"):
    fechas = pd.bdate_range(end=fin, periods=n)
    return pd.DataFrame({"AAA": np.linspace(100, 140, n), "BBB": np.linspace(100, 90, n)}, index=fechas)


def test_tras_reiniciar_cruce_medias_conserva_sus_posiciones(entorno):
    servicio = _servicio(entorno)
    estrategia = _estrategia("cruce_medias", {"rapida": 3, "lenta": 8}, frecuencia="daily")
    entorno.precios = lambda: _alcista(40, "2026-11-19")
    _pasada(entorno, servicio, estrategia, "2026-11-20")
    guardado = json.loads(servicio.ruta_estado_estrategia.read_text(encoding="utf-8"))
    assert guardado["estado"]["posiciones"]["AAA"]["fecha_entrada"] == "2026-11-19"
    # «Reinicio»: otro proceso, estrategia nueva, un día más de precios. Sin memoria, la entrada se anotaría el 11-20
    entorno.precios = lambda: _alcista(41, "2026-11-20")
    _pasada(entorno, servicio, estrategia, "2026-11-23")
    guardado = json.loads(servicio.ruta_estado_estrategia.read_text(encoding="utf-8"))
    assert guardado["estado"]["posiciones"]["AAA"]["fecha_entrada"] == "2026-11-19"
    assert guardado["estado"]["fecha"] == "2026-11-20"  # y siguió avanzando


def test_la_memoria_solo_se_restaura_con_la_misma_firma(tmp_path, caplog):
    from ypyquant.estrategias import obtener_estrategia

    ruta = tmp_path / "estado_estrategia.json"
    origen = obtener_estrategia("cruce_medias")({"rapida": 3, "lenta": 8})
    origen.estado.update(fecha="2026-11-19", modelo=object())  # un objeto en memoria no se guarda
    assert "modelo" not in serializar_estado_estrategia(origen.estado)
    from pipelines.paper_trading.estado_bot import guardar_estado_estrategia

    guardar_estado_estrategia(ruta, "firma-1", origen)
    igual, otra, ilegible = (obtener_estrategia("cruce_medias")({"rapida": 3, "lenta": 8}) for _ in range(3))
    assert restaurar_estado_estrategia(ruta, "firma-1", igual) and igual.estado == {"fecha": "2026-11-19"}
    with caplog.at_level(logging.INFO):
        assert not restaurar_estado_estrategia(ruta, "firma-2", otra) and otra.estado == {}
    ruta.write_text("{no es json", encoding="utf-8")
    assert not restaurar_estado_estrategia(ruta, "firma-1", ilegible) and ilegible.estado == {}
    assert not restaurar_estado_estrategia(tmp_path / "no_existe.json", "x", ilegible)


# ---- 5.5 fills enriquecidos e instantánea --------------------------------------------------------


def test_fills_enriquecidos_con_perfil_experimento_tipo_y_precio_de_referencia(entorno):
    servicio, estrategia, cliente = _servicio(entorno, "A", "exp-1"), _estrategia(frecuencia="daily"), entorno.cliente
    _pasada(entorno, servicio, estrategia, "2026-11-16")  # envía; aún sin ejecutar: sin fills
    assert not servicio.ruta_fills.exists()
    cliente.ejecutar("orden-1", 101.5)
    cliente.ejecutar("orden-2", 99.0)
    _pasada(entorno, servicio, estrategia, "2026-11-17")  # sincroniza al arrancar
    filas = [json.loads(linea) for linea in servicio.ruta_fills.read_text(encoding="utf-8").splitlines()]
    assert len(filas) == 2
    for fila in filas:
        assert fila["perfil"] == "A" and fila["experimento_id"] == "exp-1" and fila["tipo"] == "rebalanceo"
        assert fila["precio_referencia"] == 100.0  # último cierre cuando se decidió
        assert {"ticker", "lado", "cantidad", "precio", "comision", "fecha", "id_orden"} <= set(fila)
    assert servicio.ruta_fills.parent.name == "A"  # data/paper_trading/<perfil>/fills.jsonl
    # y se leen con el lector de la API junto a filas antiguas sin los campos nuevos
    from app.api.paper import leer_fills

    with servicio.ruta_fills.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ticker": "OLD", "lado": "compra", "cantidad": 1, "precio": 1, "id_orden": "viejo"}) + "\n")
    assert leer_fills(perfil="A", carpeta=servicio.ruta_fills.parent.parent)[0]["ticker"] == "OLD"


def test_instantanea_diaria_una_fila_por_dia_con_equity_y_posiciones(entorno):
    servicio, estrategia, cliente = _servicio(entorno), _estrategia(frecuencia="daily"), entorno.cliente
    cliente.posiciones = {"AAA": 10}
    cliente.equity = 123_456.0
    _pasada(entorno, servicio, estrategia, "2026-11-16", iteraciones=2)  # dos iteraciones el mismo día: una sola fila
    tabla = AlmacenPerfiles(entorno.raiz.paper_trading).leer_instantaneas("A")
    assert list(tabla["fecha"]) == ["2026-11-16"] and tabla["equity"].iloc[0] == 123_456.0
    assert json.loads(tabla["posiciones"].iloc[0]) == {"AAA": {"cantidad": 10.0, "precio": 0.0}}  # sin cliente de datos: precio 0
    cliente.equity = 130_000.0
    _pasada(entorno, servicio, estrategia, "2026-11-17")
    assert list(AlmacenPerfiles(entorno.raiz.paper_trading).leer_instantaneas("A")["equity"]) == [123_456.0, 130_000.0]


def test_el_precio_de_referencia_vuelve_a_cargarse_tras_reiniciar(tmp_path):
    estado = EstadoBot.cargar(tmp_path / "e.json", "f")
    estado.anadir_pendiente("o1", 101.25)
    assert EstadoBot.cargar(tmp_path / "e.json", "f").referencias == {"o1": 101.25}
    estado.quitar_pendiente("o1")
    assert EstadoBot.cargar(tmp_path / "e.json", "f").referencias == {}


# ---- vínculo desde el CLI ------------------------------------------------------------------------

CONFIG = {"estrategia": "momentum", "parametros": {"top_k": 2}, "universo": ["AAA", "BBB", "CCC", "DDD"],
          "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}


def _args(raiz, *extra):
    return crear_parser().parse_args(["--datos", str(raiz.base), *extra])


def test_cli_vincula_el_perfil_al_experimento_y_congela_su_receta(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_A_API_KEY", "PKAAAAAAAAAAAAAAAAAA1234")
    monkeypatch.setenv("ALPACA_A_SECRET_KEY", "secretoDeLaCuentaA0123456789abcdefghijkl")
    raiz = RaizDatos(base=tmp_path)
    id_experimento = AlmacenExperimentos(raiz.experimentos).registrar(ConfiguracionExperimento(**CONFIG))
    servicio, estrategia = preparar_perfil(_args(raiz, "--perfil", "A", "--experimento", id_experimento, "--modo", "diario"))
    assert estrategia.estrategia_registro == "momentum" and estrategia.tickers == ["AAA", "BBB", "CCC", "DDD"]
    assert estrategia.parametros_registro["top_k"] == 2 and estrategia.frecuencia_rebalanceo_registro == "monthly"
    assert servicio.perfil == "A" and servicio.ruta_fills.parent == raiz.paper_trading / "A"
    assert servicio.etiquetas_fills == {"perfil": "A", "tipo": "rebalanceo", "experimento_id": id_experimento}
    vinculo = AlmacenPerfiles(raiz.paper_trading).leer_vinculo("A")
    assert vinculo["origen"] == {"tipo": "experimento", "id": id_experimento} and vinculo["huella_receta"] == estrategia.huella_receta
    # sin --experimento se reanuda el vínculo guardado
    _, reanudada = preparar_perfil(_args(raiz, "--perfil", "A"))
    assert reanudada.huella_receta == estrategia.huella_receta
    assert ConfiguracionAlpaca.desde_entorno("A").clave_api.endswith("1234")


def test_cli_rechaza_combinaciones_invalidas(tmp_path):
    raiz = RaizDatos(base=tmp_path)
    for extra in (["--perfil", "a"], ["--perfil", "A"], ["--perfil", "A", "--experimento", "no-existe"],
                  ["--perfil", "A", "--estrategia", "momentum"], ["--perfil", "A", "--experimento", "x", "--configuracion", "y"]):
        with pytest.raises(SystemExit):
            preparar_perfil(_args(raiz, *extra))


def test_el_equity_inicial_lo_completa_el_bucle_al_abrir_la_cuenta(entorno):
    from ypyquant.estrategias.experimentos import ConfiguracionExperimento as Config
    from ypyquant.perfiles_alpaca.vinculo import vincular

    almacen = AlmacenPerfiles(entorno.raiz.paper_trading)
    receta = Config(**CONFIG).receta().normalizada()
    vincular(almacen, "A", receta, "experimento", "exp-1", None)
    entorno.cliente.equity = 77_000.0
    estrategia = ConfiguracionEstrategia.desde_vinculo(almacen.leer_vinculo("A"))
    entorno.precios = lambda: _precios(tickers=tuple(estrategia.tickers))
    _pasada(entorno, _servicio(entorno), estrategia, "2026-11-16")
    assert almacen.leer_vinculo("A")["equity_inicial"] == 77_000.0


# ---- seguridad: ninguna clave en el log ----------------------------------------------------------


def test_el_repr_de_la_configuracion_no_muestra_las_claves():
    texto = repr(ConfiguracionAlpaca("PKSECRETA123456789012", "supersecreto"))
    assert "PKSECRETA" not in texto and "supersecreto" not in texto


def test_un_error_del_bot_con_una_clave_sale_redactado_del_log(entorno, caplog):
    secreto = "claveSecretaDelPerfilA0123456789"
    entorno.precios = lambda: (_ for _ in ()).throw(RuntimeError(f"401 Unauthorized con {secreto}"))
    for manejador in logging.getLogger().handlers:  # caplog añade su manejador al raíz: el filtro debe verse allí también
        manejador.addFilter(FiltroRedaccion([secreto]))
    with caplog.at_level(logging.INFO):
        _pasada(entorno, _servicio(entorno), _estrategia(frecuencia="daily"), "2026-11-16")
    assert "Error en la iteración" in caplog.text and secreto not in caplog.text
