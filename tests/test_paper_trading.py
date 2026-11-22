"""
test_paper_trading.py — Tests offline del paper trading que NO necesitan el extra `trading`.

Responsabilidad:
    Portar los tests de GrowSave (`nautilus_service/tests/test_strategy.py`) que no dependen de
    nautilus_trader, y cubrir la lógica nueva: decisión de órdenes, registro de fills y config.
    Sin red ni claves de Alpaca.

Se relaciona con:
    - ypyquant.execution.nautilus: constructor_senales, ordenes, registro_fills.
    - pipelines.paper_trading: config y backtest (pesos_desde_senales).
    - test_nautilus_estrategia.py: tests que sí exigen el extra (se saltan sin él).
"""

import json
from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from pipelines.paper_trading.backtest import pesos_desde_senales
from pipelines.paper_trading.config import ConfiguracionAlpaca, ConfiguracionEstrategia
from ypyquant.execution.nautilus.constructor_senales import construir_combinador
from ypyquant.execution.nautilus.ordenes import decidir_orden
from ypyquant.execution.nautilus.registro_fills import (
    RegistroFills,
    normalizar_fill,
    parsear_fecha_alpaca,
)


def _precios_sinteticos(n=120, tickers=("AAPL", "MSFT", "GOOGL")):
    """Precios sintéticos con tendencia alcista + ruido (semilla fija)."""
    rng = np.random.default_rng(42)
    fechas = pd.date_range("2023-01-01", periods=n, freq="B")
    datos = {t: 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n))) for t in tickers}
    df = pd.DataFrame(datos, index=fechas)
    df.index.name = "date"
    return df


class TestConstruirCombinador:
    def test_combinador_por_defecto(self):
        assert construir_combinador({}) is not None

    def test_combinador_tecnico(self):
        assert construir_combinador({"generators": [{"name": "technical", "weight": 1.0}]}) is not None

    def test_generador_desconocido_usa_tecnico(self):
        assert construir_combinador({"generators": [{"name": "inexistente", "weight": 1.0}]}) is not None


class TestGenerarSenales:
    @pytest.fixture
    def combinador(self):
        return construir_combinador({"generators": [{"name": "technical", "weight": 1.0}]})

    def test_forma(self, combinador):
        senales = combinador.generate(_precios_sinteticos(n=60))
        assert senales is not None and not senales.empty
        assert set(senales.columns) == {"AAPL", "MSFT", "GOOGL"}

    def test_senales_en_rango(self, combinador):
        senales = combinador.generate(_precios_sinteticos(n=60))
        assert (senales.abs() <= 1.0).all().all(), "Las señales deben estar en [-1, +1]"

    def test_ultimas_senales_no_todas_nan(self, combinador):
        senales = combinador.generate(_precios_sinteticos(n=60))
        assert not senales.iloc[-5:].isna().all().all()

    def test_precios_vacios_no_tumban_el_proceso(self, combinador):
        try:
            combinador.generate(pd.DataFrame())
        except Exception:
            pass  # puede fallar; lo importante es que no cuelgue el proceso

    def test_pipeline_senales(self, combinador):
        senales = combinador.generate(_precios_sinteticos(n=120, tickers=("AAPL", "MSFT")))
        assert set(senales.columns) == {"AAPL", "MSFT"}
        assert (senales.abs() <= 1.0).all().all()
        assert len(senales) > 10


class TestConfiguracion:
    def test_estrategia_por_defecto(self):
        cfg = ConfiguracionEstrategia()
        assert cfg.frecuencia_rebalanceo == "daily"
        assert cfg.umbral_senal == 0.3
        assert len(cfg.tickers) > 0

    def test_estrategia_desde_json(self, tmp_path):
        ruta = tmp_path / "e.json"
        ruta.write_text(json.dumps({"nombre": "x", "tickers": ["AAPL"], "umbral_senal": 0.5}), encoding="utf-8")
        cfg = ConfiguracionEstrategia.desde_json(ruta)
        assert cfg.nombre == "x" and cfg.tickers == ["AAPL"] and cfg.umbral_senal == 0.5

    def test_ejemplo_del_repo_es_valido(self):
        from pipelines.paper_trading.config import RAIZ
        cfg = ConfiguracionEstrategia.desde_json(RAIZ / "pipelines" / "paper_trading" / "estrategia_ejemplo.json")
        assert construir_combinador(cfg.config_senales) is not None

    def test_alpaca_sin_claves_falla(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(OSError):
            ConfiguracionAlpaca.desde_entorno()

    def test_alpaca_con_claves(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        cfg = ConfiguracionAlpaca.desde_entorno()
        assert (cfg.clave_api, cfg.clave_secreta, cfg.paper) == ("k", "s", True)


class TestDecidirOrden:
    def test_compra_la_diferencia(self):
        d = decidir_orden(senal=0.8, umbral=0.3, equity=100_000, max_pos_pct=0.10, precio=100.0, cantidad_actual=20)
        assert d.lado == "compra" and d.cantidad_objetivo == 80 and d.cantidad == 60

    def test_senal_bajo_umbral_no_opera(self):
        assert decidir_orden(0.2, 0.3, 100_000, 0.10, 100.0, 0).lado is None

    def test_vende_toda_la_posicion(self):
        d = decidir_orden(-0.9, 0.3, 100_000, 0.10, 100.0, 15)
        assert d.lado == "venta" and d.cantidad == 15

    def test_senal_negativa_sin_posicion_no_opera(self):
        assert decidir_orden(-0.9, 0.3, 100_000, 0.10, 100.0, 0).lado is None

    def test_ya_en_objetivo_no_compra(self):
        assert decidir_orden(0.8, 0.3, 100_000, 0.10, 100.0, 80).lado is None

    def test_precio_invalido(self):
        assert decidir_orden(0.8, 0.3, 100_000, 0.10, 0.0, 0).lado is None


class TestPesosDesdeSenales:
    def test_solo_positivas_normalizadas(self):
        senales = pd.DataFrame({"A": [0.6], "B": [-0.2], "C": [0.2]}, index=pd.to_datetime(["2024-01-02"]))
        pesos = pesos_desde_senales(senales, pd.Timestamp("2024-01-02"))
        assert set(pesos) == {"A", "C"}
        assert pesos["A"] == pytest.approx(0.6 / 1.0)

    def test_fecha_ausente(self):
        senales = pd.DataFrame({"A": [0.6]}, index=pd.to_datetime(["2024-01-02"]))
        assert pesos_desde_senales(senales, pd.Timestamp("2025-01-01")) == {}


class TestRegistroFills:
    def test_parsear_fecha_alpaca(self):
        assert parsear_fecha_alpaca("2024-01-15T10:30:00Z").tzinfo == UTC
        assert parsear_fecha_alpaca(None) is not None
        assert parsear_fecha_alpaca("2024-06-01T14:00:00+00:00").year == 2024
        assert parsear_fecha_alpaca("basura").tzinfo is not None

    def test_normalizar_y_registrar(self, tmp_path):
        orden = {"id": "abc", "symbol": "AAPL", "side": "buy", "filled_qty": "5",
                 "filled_avg_price": "190.5", "filled_at": "2024-01-15T10:30:00Z"}
        assert normalizar_fill(orden)["lado"] == "compra"
        registro = RegistroFills(tmp_path / "sub" / "fills.jsonl")
        assert registro.registrar(orden)["cantidad"] == 5.0
        lineas = (tmp_path / "sub" / "fills.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lineas) == 1 and json.loads(lineas[0])["ticker"] == "AAPL"

    def test_fill_sin_cantidad_se_ignora(self, tmp_path):
        registro = RegistroFills(tmp_path / "fills.jsonl")
        assert registro.registrar({"id": "x", "symbol": "AAPL", "side": "sell", "filled_qty": "0"}) is None
        assert not (tmp_path / "fills.jsonl").exists()


def _ordenes_bot(pesos, equity, precios, cantidades_actuales):
    """Órdenes del Ejecutor con la política del bot (acciones enteras, umbral 1 %)."""
    from ypyquant.execution.ejecutor import POLITICA_BOT, EstadoCuenta, planificar_rebalanceo

    return planificar_rebalanceo(pesos, precios, EstadoCuenta(equity, 0.0, cantidades_actuales), POLITICA_BOT).ordenes


class TestRebalanceoPorPesos:
    """Modo registro: de pesos objetivo a órdenes (ventas primero, sin órdenes diminutas)."""

    def test_ventas_antes_que_compras(self):
        ordenes = _ordenes_bot(
            pesos={"AAA": 0.5, "BBB": 0.5}, equity=10_000, precios={"AAA": 100.0, "BBB": 50.0, "CCC": 10.0},
            cantidades_actuales={"CCC": 100, "BBB": 10},
        )
        # Ventas primero; entre las compras, la de menor diferencia de valor (BBB: 4500 < AAA: 5000)
        assert [(o.ticker, o.lado, o.cantidad) for o in ordenes] == [
            ("CCC", "venta", 100), ("BBB", "compra", 90), ("AAA", "compra", 50)]

    def test_ignora_diferencias_pequenas_pero_cierra_posiciones(self):
        ordenes = _ordenes_bot(
            pesos={"AAA": 0.5}, equity=10_000, precios={"AAA": 100.0, "BBB": 1.0},
            cantidades_actuales={"AAA": 49, "BBB": 3},
        )
        # AAA: dif 1 acción = 1 % del equity (no menor al umbral); BBB: cerrar aunque valga 0,03 %
        assert [(o.ticker, o.lado) for o in ordenes] == [("BBB", "venta"), ("AAA", "compra")]
        assert _ordenes_bot({"AAA": 0.5}, 10_000, {"AAA": 100.0}, {"AAA": 50}) == ()

    def test_estrategia_registro_desde_json(self, tmp_path):
        ruta = tmp_path / "e.json"
        ruta.write_text('{"nombre": "x", "estrategia_registro": "momentum", "parametros_registro": {"top_k": 2}}',
                        encoding="utf-8")
        est = ConfiguracionEstrategia.desde_json(ruta)
        assert est.estrategia_registro == "momentum" and est.parametros_registro == {"top_k": 2}
