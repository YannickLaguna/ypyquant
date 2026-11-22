"""
test_alpaca_margen.py — Cuentas de Alpaca con margen: declaración por perfil, vínculo, lectura de la cuenta y bot con cortos.

Responsabilidad:
    - `ALPACA_<PERFIL>_MARGEN` declara la cuenta de margen (`capacidad_declarada`); sin ella las recetas apalancadas o
      largo-corto se rechazan al vincular con un mensaje claro; con ella se admiten hasta 2x overnight.
    - `BrokerAlpaca.capacidad_margen` lee `multiplier` y `shorting_enabled` del objeto cuenta (cliente falso).
    - El bot con una estrategia largo-corto vende en corto (política del bot con `permite_cortos`) y se niega a
      arrancar si la cuenta real no es de margen. Nada de esto toca Alpaca: todo con clientes falsos.

Se relaciona con:
    src/ypyquant/perfiles_alpaca/{perfiles,vinculo}.py, src/ypyquant/execution/{margen,broker_alpaca}.py y
    pipelines/paper_trading/alpaca_vivo.py.
"""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipelines.paper_trading import alpaca_vivo
from pipelines.paper_trading.estado_bot import EstadoBot
from ypyquant.estrategias import obtener_estrategia
from ypyquant.estrategias.receta import Receta
from ypyquant.execution.broker_alpaca import BrokerAlpaca
from ypyquant.execution.broker_simulado import BrokerSimulado
from ypyquant.execution.ejecutor import VENTA
from ypyquant.execution.margen import CAPACIDAD_REG_T, SIN_MARGEN, CapacidadMargen, ReglasMargen
from ypyquant.perfiles_alpaca.almacen import AlmacenPerfiles
from ypyquant.perfiles_alpaca.perfiles import capacidad_declarada, descubrir_perfiles, variable_margen
from ypyquant.perfiles_alpaca.vinculo import RecetaNoOperable, validar_operable, vincular

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _receta(estrategia="momentum", **parametros) -> Receta:
    return Receta(estrategia=estrategia, parametros=parametros, universo=TICKERS, frecuencia_rebalanceo="monthly").normalizada()


def test_variable_y_capacidad_declarada_por_perfil():
    assert variable_margen("A") == "ALPACA_A_MARGEN" and variable_margen("DEFECTO") == "ALPACA_MARGEN"
    assert capacidad_declarada("A", {"ALPACA_A_MARGEN": "1"}) == CAPACIDAD_REG_T
    assert capacidad_declarada("A", {"ALPACA_A_MARGEN": "true"}).margen
    assert capacidad_declarada("A", {"ALPACA_B_MARGEN": "1"}) == SIN_MARGEN  # solo el perfil que lo declara
    assert capacidad_declarada("DEFECTO", {"ALPACA_MARGEN": "1"}).margen
    perfiles = descubrir_perfiles({"ALPACA_A_API_KEY": "k1234", "ALPACA_A_SECRET_KEY": "s", "ALPACA_A_MARGEN": "1",
                                   "ALPACA_B_API_KEY": "k5678", "ALPACA_B_SECRET_KEY": "s"})
    assert [(p.nombre, p.a_dict()["margen"]) for p in perfiles] == [("A", True), ("B", False)]


def test_recetas_apalancadas_y_largo_corto_exigen_cuenta_de_margen():
    largo_corto = _receta(exposicion_larga=1.3, exposicion_corta=0.3)
    apalancada = _receta("bayes_stein", volatilidad_objetivo=0.3, apalancamiento_maximo=1.5)
    for receta in (largo_corto, apalancada):
        with pytest.raises(RecetaNoOperable, match="cuenta de margen"):
            validar_operable(receta)  # sin capacidad declarada: rechazo como antes
        assert validar_operable(receta, CAPACIDAD_REG_T) == TICKERS
    assert validar_operable(_receta()) == TICKERS  # largo y sin apalancar: cualquier cuenta


def test_limites_de_la_cuenta_multiplicador_y_cortos():
    receta = _receta("bayes_stein", volatilidad_objetivo=0.3, apalancamiento_maximo=2.0)
    with pytest.raises(RecetaNoOperable, match="supera"):
        validar_operable(receta, CapacidadMargen(margen=True, multiplicador=1.5, cortos=True))
    assert validar_operable(receta, CapacidadMargen(margen=True, multiplicador=4.0, cortos=False))  # 4x es intradía: se acota a 2x
    sin_cortos = CapacidadMargen(margen=True, multiplicador=2.0, cortos=False)
    with pytest.raises(RecetaNoOperable, match="ventas en corto"):
        validar_operable(_receta(exposicion_larga=1.3, exposicion_corta=0.3), sin_cortos)


def test_vincular_con_cuenta_de_margen_guarda_el_vinculo(tmp_path):
    receta = _receta(exposicion_larga=1.3, exposicion_corta=0.3)
    with pytest.raises(RecetaNoOperable):
        vincular(AlmacenPerfiles(tmp_path), "A", receta, "experimento", "x", None)
    assert AlmacenPerfiles(tmp_path).leer_vinculo("A") is None
    vinculo, cambio = vincular(AlmacenPerfiles(tmp_path), "A", receta, "experimento", "x", None, capacidad=CAPACIDAD_REG_T)
    assert cambio and vinculo["receta"]["parametros"]["exposicion_corta"] == 0.3


def test_broker_alpaca_lee_multiplicador_y_shorting_de_la_cuenta():
    def cliente(**campos):
        return SimpleNamespace(get_account=lambda: SimpleNamespace(**campos))

    assert BrokerAlpaca(cliente(multiplier="2", shorting_enabled=True)).capacidad_margen() == CapacidadMargen(True, 2.0, True)
    assert BrokerAlpaca(cliente(multiplier="1", shorting_enabled=False)).capacidad_margen() == SIN_MARGEN
    assert BrokerAlpaca(cliente(multiplier="4", shorting_enabled=True)).capacidad_margen().apalancamiento_permitido == 2.0
    assert not BrokerAlpaca(cliente()).capacidad_margen().margen  # cuenta sin los campos: sin margen


def _precios(n=400) -> pd.DataFrame:
    indice = pd.bdate_range("2025-01-02", periods=n)
    generador = np.random.default_rng(5)
    return pd.DataFrame({t: 100 * np.exp(np.cumsum(generador.normal(0.0006 * (i - 2.5), 0.01, n))) for i, t in enumerate(TICKERS)},
                        index=indice)


def test_el_bot_con_estrategia_largo_corto_vende_en_corto(tmp_path):
    estrategia = obtener_estrategia("momentum")({"top_k": 2, "meses_retorno": 6, "exposicion_larga": 1.3, "exposicion_corta": 0.3})
    precios = _precios()
    reglas = ReglasMargen(apalancamiento_maximo=1.6, permite_cortos=True)
    broker = BrokerSimulado(100_000.0, precios={t: float(precios[t].iloc[-1]) for t in TICKERS}, reglas_margen=reglas)
    estado = EstadoBot.cargar(tmp_path / "estado_bot.json", "firma")
    pesos = alpaca_vivo.rebalancear_con_estrategia_registro(broker, estrategia, precios, True, estado, "monthly",
                                                            hoy=date(2026, 7, 1))
    assert pesos is not None and any(w < 0 for w in pesos.values())
    ventas_en_corto = [t for t, lado, _ in broker.enviadas if lado == VENTA]
    assert len(ventas_en_corto) == 2 and all(broker.posiciones[t] < 0 for t in ventas_en_corto)
    assert estado.ultimo_rebalanceo == date(2026, 7, 1)
    assert broker.estado_cuenta().equity == pytest.approx(100_000.0, rel=0.01)


def test_el_bot_se_niega_a_arrancar_si_la_cuenta_real_no_es_de_margen(tmp_path, monkeypatch):
    """Aunque el .env declare margen, la cuenta REAL (multiplier 1, sin shorting) manda: no se opera una estrategia con cortos."""
    from pipelines.paper_trading import config as cfg

    cuenta = SimpleNamespace(equity="100000", cash="100000", multiplier="1", shorting_enabled=False)
    cliente = SimpleNamespace(get_account=lambda: cuenta, get_all_positions=list)
    monkeypatch.setattr(alpaca_vivo, "crear_clientes", lambda servicio: (cliente, None))
    servicio = cfg.ConfiguracionServicio(
        alpaca=cfg.ConfiguracionAlpaca("k", "s"), segundos_entre_iteraciones=1, dias_historico=252,
        ruta_fills=tmp_path / "fills.jsonl", ruta_estado=tmp_path / "estado_bot.json")
    estrategia = cfg.ConfiguracionEstrategia(
        nombre="momentum", tickers=TICKERS, estrategia_registro="momentum",
        parametros_registro={"exposicion_larga": 1.3, "exposicion_corta": 0.3})
    with pytest.raises(RuntimeError, match="ventas en corto"):
        alpaca_vivo.ejecutar_vivo(servicio, estrategia, alpaca_vivo.ControlParada(), max_iteraciones=1)
