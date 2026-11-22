"""
test_paper_vivo.py — Tests offline del bucle de paper trading en modo registro (sin red ni claves).

Responsabilidad:
    Cubrir con un cliente de Alpaca falso lo que antes fallaba en silencio: historia descargada
    suficiente para la estrategia (y aviso si aun así falta), registro de los fills reales (sin
    duplicados), rebalanceo solo cuando toca según la frecuencia (persistida entre reinicios) y
    no duplicar órdenes cuando ya hay una abierta sobre el mismo ticker.

Se relaciona con:
    - pipelines/paper_trading/alpaca_vivo.py, precios.py, estado_bot.py, __main__.py y config.py.
    - ypyquant.estrategias (sesiones_necesarias de cada estrategia) y ypyquant.execution.nautilus.
    - app/api/paper.py y rutas_paper.py (parámetros `dias_historico` y `frecuencia_rebalanceo` del bot).
"""

import json
import logging
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipelines.paper_trading import alpaca_vivo
from pipelines.paper_trading.__main__ import crear_parser
from pipelines.paper_trading.estado_bot import EstadoBot, periodo_de_rebalanceo, toca_rebalancear
from pipelines.paper_trading.precios import (
    avisar_historia_insuficiente,
    dias_naturales_para_sesiones,
    resolver_dias_historico,
)
from ypyquant.estrategias import obtener_estrategia
from ypyquant.estrategias.base import MARGEN_SESIONES
from ypyquant.execution.broker_alpaca import BrokerAlpaca
from ypyquant.execution.ejecutor import POLITICA_BOT, EstadoCuenta, planificar_rebalanceo
from ypyquant.execution.nautilus.registro_fills import RegistroFills

pytest.importorskip("alpaca")  # extra `trading`: se construyen MarketOrderRequest reales


# ---- utilidades de prueba ---------------------------------------------------------------------


class ClienteAlpacaFalso:
    """
    Imita lo mínimo de `TradingClient`: cuenta, posiciones, órdenes abiertas, envío y consulta por id.

    Las órdenes enviadas nacen en estado `new` con `filled_qty` 0 (como en Alpaca); `ejecutar` las
    llena al precio dado. `fallar_ordenes_abiertas` simula un error al consultar las abiertas.
    """

    def __init__(self, equity=100_000.0, posiciones=None, abierto=True):
        self.equity = equity
        self.posiciones = posiciones or {}
        self.abierto = abierto
        self.enviadas: list = []
        self.ordenes: dict[str, SimpleNamespace] = {}
        self.fallar_ordenes_abiertas = False

    def get_clock(self):
        return SimpleNamespace(is_open=self.abierto)

    def get_account(self):
        return SimpleNamespace(equity=str(self.equity), cash="0", account_number="PA-TEST")

    def get_all_positions(self):
        return [SimpleNamespace(symbol=t, qty=str(q)) for t, q in self.posiciones.items()]

    def get_orders(self, _peticion):
        if self.fallar_ordenes_abiertas:
            raise ConnectionError("sin red")
        return [o for o in self.ordenes.values() if o.status.value in {"new", "accepted", "partially_filled"}]

    def submit_order(self, peticion):
        identificador = f"orden-{len(self.ordenes) + 1}"
        orden = SimpleNamespace(
            id=identificador, symbol=peticion.symbol, side=peticion.side, qty=str(peticion.qty),
            status=SimpleNamespace(value="new"), filled_qty="0", filled_avg_price=None, filled_at=None,
        )
        self.ordenes[identificador] = orden
        self.enviadas.append(peticion)
        return orden

    def get_order_by_id(self, identificador):
        return self.ordenes[str(identificador)]

    def ejecutar(self, identificador, precio, cantidad=None, estado="filled"):
        """Marca la orden como ejecutada (total por defecto) al `precio`."""
        orden = self.ordenes[identificador]
        orden.status = SimpleNamespace(value=estado)
        orden.filled_qty = str(cantidad if cantidad is not None else orden.qty)
        orden.filled_avg_price = str(precio)
        orden.filled_at = pd.Timestamp("2026-11-20 15:00", tz="UTC").to_pydatetime()


def _precios(tickers=("AAA", "BBB"), n=30, precio=100.0):
    """Cierres planos fecha x ticker (la estrategia de pesos iguales no depende de ellos)."""
    fechas = pd.bdate_range(end="2026-11-19", periods=n)
    return pd.DataFrame({t: np.full(n, precio) for t in tickers}, index=fechas)


def _estado(tmp_path, firma="pesos_iguales|{}|AAA,BBB"):
    return EstadoBot.cargar(tmp_path / "estado_bot.json", firma)


# ---- 1. historia suficiente -------------------------------------------------------------------


class TestHistoriaSuficiente:
    def test_momentum_declara_sus_sesiones_segun_parametros(self):
        defecto = obtener_estrategia("momentum")({})
        assert defecto.sesiones_necesarias() == 12 * 21 + 1 + MARGEN_SESIONES  # 12-1 sobre DIAS_MES=21
        corto = obtener_estrategia("momentum")({"meses_retorno": 6})
        assert corto.sesiones_necesarias() == 6 * 21 + 1 + MARGEN_SESIONES

    @pytest.mark.parametrize("nombre", ["bayes_stein", "michaud", "bl_prior", "robusta_elipsoide", "robusta_caja"])
    def test_estimadores_declaran_su_ventana(self, nombre):
        estrategia = obtener_estrategia(nombre)({"ventana_dias": 500})
        assert estrategia.sesiones_necesarias() == 500 + 1 + MARGEN_SESIONES

    def test_preset_declara_su_ventana_y_pesos_iguales_no_exige_historia(self):
        assert obtener_estrategia("preset_ypyquant")({"ventana_dias": 300}).sesiones_necesarias() == 300 + 1 + MARGEN_SESIONES
        assert obtener_estrategia("pesos_iguales")({}).sesiones_necesarias() == 0

    def test_sesiones_a_dias_naturales_con_margen(self):
        # 253 sesiones ~ 1 año de calendario + festivos; nunca menos de lo necesario
        assert dias_naturales_para_sesiones(0) == 0
        assert dias_naturales_para_sesiones(253) > 365
        assert dias_naturales_para_sesiones(767) > 1_100  # ventana de 756 sesiones de los estimadores del TFM

    def test_resolver_dias_usa_el_mayor_entre_estrategia_y_configurado(self):
        necesarios = dias_naturales_para_sesiones(263)
        assert resolver_dias_historico(263, 252) == necesarios  # el env por defecto se queda corto
        assert resolver_dias_historico(263, 2_000) == 2_000  # el configurado explícito, si es mayor, se respeta
        assert resolver_dias_historico(0, 252) == 252  # estrategia que no declara historia: el configurado

    def test_aviso_claro_si_faltan_sesiones(self, caplog):
        precios = _precios(n=183)
        with caplog.at_level(logging.WARNING):
            assert avisar_historia_insuficiente(precios, 263, "momentum") is True
        mensaje = caplog.records[-1].getMessage()
        assert caplog.records[-1].levelno == logging.WARNING
        assert "momentum" in mensaje and "183" in mensaje and "263" in mensaje and "pesos iguales" in mensaje

    def test_aviso_por_ticker_con_poca_historia(self, caplog):
        precios = _precios(n=300)
        precios.loc[precios.index[:200], "BBB"] = np.nan  # BBB solo tiene 100 sesiones
        with caplog.at_level(logging.WARNING):
            assert avisar_historia_insuficiente(precios, 263, "momentum") is True
        assert "BBB" in caplog.records[-1].getMessage()

    def test_sin_aviso_si_hay_historia_de_sobra(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert avisar_historia_insuficiente(_precios(n=300), 263, "momentum") is False
        assert not caplog.records

    def test_cli_acepta_dias_y_frecuencia(self):
        argumentos = crear_parser().parse_args(["--estrategia", "momentum", "--dias-historico", "500", "--frecuencia", "weekly"])
        assert argumentos.dias_historico == 500 and argumentos.frecuencia == "weekly"
        defecto = crear_parser().parse_args([])
        assert defecto.dias_historico is None and defecto.frecuencia is None


# ---- 2. fills reales --------------------------------------------------------------------------


class TestFillsReales:
    def _enviar(self, cliente, estado, tmp_path):
        """Una iteración de rebalanceo: devuelve los pesos y deja las órdenes enviadas como pendientes."""
        estrategia = obtener_estrategia("pesos_iguales")({})
        return alpaca_vivo.rebalancear_con_estrategia_registro(
            BrokerAlpaca(cliente), estrategia, _precios(), True, estado, "daily", hoy=date(2026, 11, 20))

    def test_las_ordenes_enviadas_quedan_pendientes_y_no_se_registran_con_cantidad_cero(self, tmp_path):
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        self._enviar(cliente, estado, tmp_path)
        assert len(cliente.enviadas) == 2 and set(estado.ordenes_pendientes) == {"orden-1", "orden-2"}
        registro = RegistroFills(tmp_path / "fills.jsonl")
        # Aún sin ejecutar: no se registra nada y siguen pendientes
        assert alpaca_vivo.sincronizar_fills(BrokerAlpaca(cliente), registro, estado, reintentos=0, espera_s=0) == 0
        assert not (tmp_path / "fills.jsonl").exists() and len(estado.ordenes_pendientes) == 2

    def test_registra_el_fill_real_cuando_se_ejecuta_y_no_duplica(self, tmp_path):
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        self._enviar(cliente, estado, tmp_path)
        cliente.ejecutar("orden-1", 101.5)
        cliente.ejecutar("orden-2", 99.0)
        registro = RegistroFills(tmp_path / "fills.jsonl")
        assert alpaca_vivo.sincronizar_fills(BrokerAlpaca(cliente), registro, estado, reintentos=0, espera_s=0) == 2
        filas = [json.loads(linea) for linea in (tmp_path / "fills.jsonl").read_text(encoding="utf-8").splitlines()]
        assert {f["id_orden"] for f in filas} == {"orden-1", "orden-2"}
        assert {f["precio"] for f in filas} == {101.5, 99.0} and all(f["cantidad"] == 500 for f in filas)
        assert estado.ordenes_pendientes == []
        # Segunda sincronización y recarga desde disco: nada nuevo
        assert alpaca_vivo.sincronizar_fills(BrokerAlpaca(cliente), registro, estado, reintentos=0, espera_s=0) == 0
        assert len((tmp_path / "fills.jsonl").read_text(encoding="utf-8").splitlines()) == 2
        assert _estado(tmp_path).ordenes_pendientes == []

    def test_parcial_sigue_pendiente_y_cancelada_con_fill_parcial_se_registra(self, tmp_path):
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        self._enviar(cliente, estado, tmp_path)
        registro = RegistroFills(tmp_path / "fills.jsonl")
        cliente.ejecutar("orden-1", 100.0, cantidad=100, estado="partially_filled")
        cliente.ejecutar("orden-2", 100.0, cantidad=200, estado="canceled")
        assert alpaca_vivo.sincronizar_fills(BrokerAlpaca(cliente), registro, estado, reintentos=0, espera_s=0) == 1
        assert estado.ordenes_pendientes == ["orden-1"]  # la parcial aún puede completarse
        cliente.ordenes["orden-1"].status = SimpleNamespace(value="canceled")
        assert alpaca_vivo.sincronizar_fills(BrokerAlpaca(cliente), registro, estado, reintentos=0, espera_s=0) == 1
        assert estado.ordenes_pendientes == []

    def test_rechazada_sin_fill_se_descarta_sin_registrar(self, tmp_path):
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        self._enviar(cliente, estado, tmp_path)
        for identificador in ("orden-1", "orden-2"):
            cliente.ordenes[identificador].status = SimpleNamespace(value="rejected")
        registro = RegistroFills(tmp_path / "fills.jsonl")
        assert alpaca_vivo.sincronizar_fills(BrokerAlpaca(cliente), registro, estado, reintentos=0, espera_s=0) == 0
        assert estado.ordenes_pendientes == [] and not (tmp_path / "fills.jsonl").exists()

    def test_error_de_consulta_deja_la_orden_pendiente(self, tmp_path, caplog):
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        self._enviar(cliente, estado, tmp_path)
        cliente.get_order_by_id = lambda _id: (_ for _ in ()).throw(ConnectionError("caído"))
        with caplog.at_level(logging.WARNING):
            alpaca_vivo.sincronizar_fills(BrokerAlpaca(cliente), RegistroFills(tmp_path / "fills.jsonl"), estado, reintentos=0, espera_s=0)
        assert len(estado.ordenes_pendientes) == 2 and "caído" in caplog.text

    def test_reintenta_hasta_que_se_ejecuten(self, tmp_path):
        """Con reintentos > 0 espera `espera_s` entre consultas (aquí 0 s) y registra al llenarse."""
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        self._enviar(cliente, estado, tmp_path)
        consultas = {"n": 0}
        original = cliente.get_order_by_id

        def consulta(identificador):
            consultas["n"] += 1
            if consultas["n"] == 3:  # al tercer intento ya están ejecutadas
                cliente.ejecutar("orden-1", 100.0)
                cliente.ejecutar("orden-2", 100.0)
            return original(identificador)

        cliente.get_order_by_id = consulta
        registro = RegistroFills(tmp_path / "fills.jsonl")
        assert alpaca_vivo.sincronizar_fills(BrokerAlpaca(cliente), registro, estado, reintentos=3, espera_s=0) == 2

    def test_registro_fills_no_duplica_el_mismo_id_de_orden(self, tmp_path):
        registro = RegistroFills(tmp_path / "fills.jsonl")
        orden = {"id": "x1", "symbol": "AAPL", "side": "buy", "filled_qty": "5", "filled_avg_price": "190", "filled_at": None}
        assert registro.registrar(orden) is not None
        assert registro.registrar(orden) is None
        assert len((tmp_path / "fills.jsonl").read_text(encoding="utf-8").splitlines()) == 1


# ---- 3. frecuencia de rebalanceo --------------------------------------------------------------


class TestFrecuenciaDeRebalanceo:
    @pytest.mark.parametrize(("frecuencia", "ultimo", "hoy", "toca"), [
        ("daily", None, date(2026, 11, 20), True),  # nunca rebalanceado
        ("daily", date(2026, 11, 20), date(2026, 11, 20), False),  # mismo día
        ("daily", date(2026, 11, 19), date(2026, 11, 20), True),
        ("weekly", date(2026, 11, 16), date(2026, 11, 20), False),  # misma semana ISO
        ("weekly", date(2026, 11, 13), date(2026, 11, 16), True),  # semana siguiente
        ("monthly", date(2026, 11, 2), date(2026, 11, 30), False),  # mismo mes
        ("monthly", date(2026, 10, 30), date(2026, 11, 2), True),  # mes siguiente
        ("monthly", date(2025, 11, 2), date(2026, 11, 2), True),  # mismo mes, otro año
        ("quarterly", date(2026, 10, 1), date(2026, 12, 31), False),
        ("quarterly", date(2026, 9, 30), date(2026, 10, 1), True),
        ("monthly", date(2026, 11, 20), date(2026, 11, 1), False),  # reloj atrasado: no se rebalancea
    ])
    def test_toca_rebalancear(self, frecuencia, ultimo, hoy, toca):
        assert toca_rebalancear(frecuencia, ultimo, hoy) is toca

    def test_frecuencia_desconocida_falla(self):
        with pytest.raises(ValueError, match="frecuencia"):
            periodo_de_rebalanceo("anual", date(2026, 1, 1))

    def test_estado_persiste_y_se_reinicia_si_cambia_la_estrategia(self, tmp_path):
        estado = _estado(tmp_path)
        estado.marcar_rebalanceo(date(2026, 11, 20))
        assert _estado(tmp_path).ultimo_rebalanceo == date(2026, 11, 20)  # sobrevive al reinicio
        assert _estado(tmp_path, firma="momentum|{}|AAA,BBB").ultimo_rebalanceo is None  # otra estrategia: no hereda

    def test_estado_tolera_fichero_corrupto(self, tmp_path):
        (tmp_path / "estado_bot.json").write_text("{no es json", encoding="utf-8")
        estado = _estado(tmp_path)
        assert estado.ultimo_rebalanceo is None and estado.ordenes_pendientes == []

    def test_solo_rebalancea_cuando_toca_y_recuerda_la_fecha(self, tmp_path, caplog):
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        estrategia = obtener_estrategia("pesos_iguales")({})
        rebalancear = alpaca_vivo.rebalancear_con_estrategia_registro
        # Primera vez: toca, envía y marca
        assert rebalancear(BrokerAlpaca(cliente), estrategia, _precios(), True, estado, "monthly", hoy=date(2026, 11, 2)) is not None
        assert len(cliente.enviadas) == 2 and estado.ultimo_rebalanceo == date(2026, 11, 2)
        # Mismo mes, otro día e incluso con la cartera desviada: no toca, solo loguea
        cliente.posiciones = {"AAA": 1, "BBB": 900}
        with caplog.at_level(logging.INFO):
            assert rebalancear(BrokerAlpaca(cliente), estrategia, _precios(), True, estado, "monthly", hoy=date(2026, 11, 20)) is None
        assert len(cliente.enviadas) == 2 and "no toca" in caplog.text.lower()
        # Tras reiniciar el proceso (estado recargado de disco) sigue sin tocar
        assert rebalancear(BrokerAlpaca(cliente), estrategia, _precios(), True, _estado(tmp_path), "monthly", hoy=date(2026, 11, 25)) is None
        # Mes siguiente: toca de nuevo
        assert rebalancear(BrokerAlpaca(cliente), estrategia, _precios(), True, _estado(tmp_path), "monthly", hoy=date(2026, 12, 1)) is not None

    def test_mercado_cerrado_no_envia_ni_marca(self, tmp_path):
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        estrategia = obtener_estrategia("pesos_iguales")({})
        alpaca_vivo.rebalancear_con_estrategia_registro(BrokerAlpaca(cliente), estrategia, _precios(), False, estado, "daily", hoy=date(2026, 11, 20))
        assert cliente.enviadas == [] and estado.ultimo_rebalanceo is None

    def test_si_falla_un_envio_no_marca_para_reintentar(self, tmp_path):
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        estrategia = obtener_estrategia("pesos_iguales")({})
        envio = cliente.submit_order
        cliente.submit_order = lambda peticion: (_ for _ in ()).throw(RuntimeError("rechazada")) if peticion.symbol == "BBB" else envio(peticion)
        alpaca_vivo.rebalancear_con_estrategia_registro(BrokerAlpaca(cliente), estrategia, _precios(), True, estado, "daily", hoy=date(2026, 11, 20))
        assert len(cliente.enviadas) == 1 and estado.ultimo_rebalanceo is None


# ---- 4. órdenes abiertas ----------------------------------------------------------------------


class TestOrdenesAbiertas:
    def test_decision_ignora_los_tickers_bloqueados(self):
        cuenta, precios = EstadoCuenta(10_000, 0.0, {}), {"AAA": 100.0, "BBB": 50.0}
        sin_bloqueo = planificar_rebalanceo({"AAA": 0.5, "BBB": 0.5}, precios, cuenta, POLITICA_BOT)
        assert [o.ticker for o in sin_bloqueo.ordenes] == ["AAA", "BBB"]
        con_bloqueo = planificar_rebalanceo({"AAA": 0.5, "BBB": 0.5}, precios, cuenta, POLITICA_BOT,
                                            tickers_bloqueados={"AAA"})
        assert [o.ticker for o in con_bloqueo.ordenes] == ["BBB"]
        assert [(o.ticker, o.motivo) for o in con_bloqueo.omitidas] == [("AAA", "bloqueado")]

    def test_no_duplica_una_orden_ya_abierta(self, tmp_path, caplog):
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        estrategia = obtener_estrategia("pesos_iguales")({})
        rebalancear = alpaca_vivo.rebalancear_con_estrategia_registro
        rebalancear(BrokerAlpaca(cliente), estrategia, _precios(), True, estado, "daily", hoy=date(2026, 11, 19))
        assert len(cliente.enviadas) == 2  # ambas siguen en estado `new` (abiertas)
        # Al día siguiente toca de nuevo, pero las dos órdenes siguen abiertas: no se envía ninguna más
        with caplog.at_level(logging.INFO):
            rebalancear(BrokerAlpaca(cliente), estrategia, _precios(), True, estado, "daily", hoy=date(2026, 11, 20))
        assert len(cliente.enviadas) == 2 and "orden abierta" in caplog.text.lower()

    def test_solo_se_bloquea_el_ticker_con_orden_abierta(self, tmp_path):
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        estrategia = obtener_estrategia("pesos_iguales")({})
        cliente.ordenes["previa"] = SimpleNamespace(id="previa", symbol="AAA", status=SimpleNamespace(value="accepted"))
        alpaca_vivo.rebalancear_con_estrategia_registro(BrokerAlpaca(cliente), estrategia, _precios(), True, estado, "daily", hoy=date(2026, 11, 20))
        assert [p.symbol for p in cliente.enviadas] == ["BBB"]

    def test_si_no_se_pueden_consultar_las_abiertas_no_se_envia_nada(self, tmp_path, caplog):
        cliente, estado = ClienteAlpacaFalso(), _estado(tmp_path)
        cliente.fallar_ordenes_abiertas = True
        estrategia = obtener_estrategia("pesos_iguales")({})
        with caplog.at_level(logging.WARNING):
            alpaca_vivo.rebalancear_con_estrategia_registro(BrokerAlpaca(cliente), estrategia, _precios(), True, estado, "daily", hoy=date(2026, 11, 20))
        assert cliente.enviadas == [] and estado.ultimo_rebalanceo is None and "órdenes abiertas" in caplog.text


# ---- 5. la historia que se descarga sale de la estrategia -------------------------------------


def test_ejecutar_vivo_descarga_la_historia_que_necesita_la_estrategia(tmp_path, monkeypatch, caplog):
    """Momentum necesita ~263 sesiones: con el mínimo de 252 días del env, `descargar_precios` debe pedir más."""
    from pipelines.paper_trading import config as cfg

    pedidos = {}

    def descarga_falsa(tickers, dias):
        pedidos["dias"] = dias
        return _precios(tickers=tuple(tickers), n=100)  # insuficiente a propósito: debe avisar

    monkeypatch.setattr(alpaca_vivo, "descargar_precios", descarga_falsa)
    cliente = ClienteAlpacaFalso(abierto=False)
    monkeypatch.setattr(alpaca_vivo, "crear_clientes", lambda servicio: (cliente, None))
    servicio = cfg.ConfiguracionServicio(
        alpaca=cfg.ConfiguracionAlpaca("k", "s"), segundos_entre_iteraciones=1, dias_historico=252,
        ruta_fills=tmp_path / "fills.jsonl", ruta_estado=tmp_path / "estado_bot.json")
    estrategia = cfg.ConfiguracionEstrategia(nombre="momentum", tickers=["AAA", "BBB"], estrategia_registro="momentum")
    with caplog.at_level(logging.INFO):
        alpaca_vivo.ejecutar_vivo(servicio, estrategia, alpaca_vivo.ControlParada(), max_iteraciones=1)
    assert pedidos["dias"] == dias_naturales_para_sesiones(12 * 21 + 1 + MARGEN_SESIONES) > 252
    avisos = [r for r in caplog.records if r.levelno == logging.WARNING and "sesiones" in r.getMessage()]
    assert avisos, "debe avisar de que la historia es insuficiente"
