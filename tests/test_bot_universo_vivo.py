"""
test_bot_universo_vivo.py — El bot de Alpaca con el universo `sp500_pit`: miembros vigentes el día del rebalanceo.

Responsabilidad (cliente Alpaca falso, reloj inyectado y membresías sintéticas; sin red ni claves):
    - el vínculo de una receta `sp500_pit` ya no se rechaza: guarda `universo_dinamico` y los tickers vigentes hoy;
    - el bot rebalancea hacia los miembros vigentes de la fecha: descarga precios de los vigentes más lo que hay en
      cartera, y un valor que salió del índice recibe una orden de venta en el rebalanceo (no antes), con aviso;
    - avisos de membresías antiguas (>30 días) o con número de miembros anómalo en el log;
    - la firma del estado no depende de los miembros de cada día (solo del nombre del universo).

Se relaciona con: pipelines/paper_trading (alpaca_vivo, config), ypyquant.perfiles_alpaca.vinculo,
ypyquant.data.universos y tests/test_bot_experimento.py (misma fábrica de entorno falso).
"""

import logging

import pytest

from pipelines.paper_trading import alpaca_vivo
from pipelines.paper_trading.config import ConfiguracionEstrategia
from tests.test_bot_experimento import _pasada, _servicio, entorno, sin_claves  # noqa: F401 - fixtures compartidas
from tests.test_paper_vivo import _precios
from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.perfiles_alpaca.vinculo import resolver_tickers

pytest.importorskip("alpaca")


def _m(alta, baja=None):
    return [{"index": "SP500", "date_added": alta, "date_removed": baja}]


# BBB sale del índice el 2026-11-10; CCC entra el 2026-11-12
MEMBRESIAS = {"AAA": _m("2010-01-04"), "BBB": _m("2015-01-01", "2026-11-10"), "CCC": _m("2026-11-12")}


def _estrategia_pit(frecuencia="weekly"):
    return ConfiguracionEstrategia(nombre="pesos_iguales", tickers=["AAA", "BBB"], estrategia_registro="pesos_iguales",
                                   parametros_registro={}, frecuencia_rebalanceo_registro=frecuencia,
                                   huella_receta="h1", origen={"tipo": "experimento", "id": "exp-1"}, universo_dinamico=True)


def _config_pit():
    return ConfiguracionExperimento(estrategia="pesos_iguales", parametros={}, universo="sp500_pit",
                                    fecha_inicio="2020-01-01", fecha_fin="2020-12-31")


def test_resolver_tickers_de_sp500_pit_da_los_vigentes_hoy(monkeypatch):
    monkeypatch.setattr("ypyquant.perfiles_alpaca.vinculo.cargar_membresias_sp500", lambda: MEMBRESIAS)
    receta = _config_pit().receta().normalizada()
    assert resolver_tickers(receta, "2026-11-11") == ["AAA"]
    assert resolver_tickers(receta, "2026-11-12") == ["AAA", "CCC"]


def test_la_firma_no_depende_de_los_miembros_de_cada_dia():
    a, b = _estrategia_pit(), _estrategia_pit()
    b.tickers = ["AAA", "CCC", "DDD"]
    assert alpaca_vivo._firma_estrategia(a) == alpaca_vivo._firma_estrategia(b)
    assert "sp500_pit" in alpaca_vivo._firma_estrategia(a)


def test_el_bot_descarga_los_vigentes_mas_la_cartera_y_liquida_al_que_salio(entorno, caplog, monkeypatch):  # noqa: F811
    descargados = []
    monkeypatch.setattr(alpaca_vivo, "descargar_precios",
                        lambda tickers, dias: descargados.append(sorted(tickers)) or _precios(("AAA", "BBB", "CCC"), n=330))
    entorno.cliente.posiciones = {"BBB": 500}  # en cartera, pero ya fuera del índice el 2026-11-16
    servicio, estrategia = _servicio(entorno), _estrategia_pit()
    with caplog.at_level(logging.INFO):
        _pasada(entorno, servicio, estrategia, "2026-11-16", membresias=MEMBRESIAS)
    assert descargados == [["AAA", "BBB", "CCC"]]  # vigentes (AAA, CCC) + la posición que sigue en cartera (BBB)
    ventas = [o for o in entorno.cliente.enviadas if "BBB" in str(o)]
    assert len(ventas) == 1  # BBB, que ya no es miembro, se vende en este rebalanceo
    textos = " ".join(r.getMessage() for r in caplog.records)
    pesos = [r.getMessage() for r in caplog.records if "Pesos objetivo" in r.getMessage()]
    assert len(pesos) == 1 and "'AAA': 0.5" in pesos[0] and "'CCC': 0.5" in pesos[0] and "BBB" not in pesos[0]
    assert "BBB salió del S&P 500" in textos
    assert "miembros vigentes" in textos  # 2 miembros frente a ~500
    assert "antigüedad" not in textos  # la última membresía (2026-11-12) tiene solo 4 días
