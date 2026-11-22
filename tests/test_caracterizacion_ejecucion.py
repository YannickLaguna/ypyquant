"""
test_caracterizacion_ejecucion.py — Fija el comportamiento ACTUAL de la conversión pesos -> órdenes.

Responsabilidad:
    Tests de caracterización (escritos ANTES de introducir el módulo Ejecutor) de las dos conversiones
    que existían por separado, para que la unificación no cambie ningún resultado:
    - motor de backtest (`BacktestEngine._execute_rebalance`): fracciones, umbral de 10 unidades de
      divisa, ventas antes que compras, normalización si la suma se desvía más de 0.01, pseudo-ticker
      CASH (se salta por no tener precio), posiciones sin precio, rotación;
    - bot (`decidir_ordenes_rebalanceo`, hoy Ejecutor con POLITICA_BOT): acciones enteras (floor), umbral del 1 % del equity, cierre de
      posiciones aunque estén bajo el umbral, tickers bloqueados, CASH/sin precio descartados, suma de
      pesos distinta de 1 (el bot NO normaliza).
    Los valores numéricos están fijados con el código previo; si alguno cambia, el refactor ha alterado
    resultados.

Se relaciona con:
    - src/ypyquant/portfolio/backtest_engine.py (motor) y src/ypyquant/ejecucion/ejecutor.py (Ejecutor).
    - Único cambio observable del bot al unificar: dentro de las compras (y de las ventas) se envía primero la de
      menor diferencia de valor (regla del motor), no el orden alfabético; las cantidades no cambian.
    - tests/test_backtest_motor.py, tests/test_paper_vivo.py y tests/test_paper_trading.py (resto de casos).
"""

from dataclasses import replace

import pandas as pd
import pytest

from ypyquant.execution.ejecutor import POLITICA_BOT, EstadoCuenta, planificar_rebalanceo
from ypyquant.portfolio.backtest_engine import BacktestConfig, BacktestEngine

DIA = pd.Timestamp("2023-01-02")


def _motor(comision=0.001, slippage=0.001, capital=10_000.0, posiciones=None, efectivo=None):
    """Motor con estado inicial a mano (mismos pasos que `run_backtest`): capital en efectivo y posiciones dadas."""
    cfg = BacktestConfig(start_date="2023-01-02", end_date="2023-12-29", initial_capital=capital,
                         rebalance_frequency="monthly", commission_rate=comision, slippage_pct=slippage)
    motor = BacktestEngine(cfg)
    motor._reset_estado()
    motor.positions = {t: {"quantity": q, "avg_price": p} for t, (q, p) in (posiciones or {}).items()}
    motor.cash = capital if efectivo is None else efectivo
    motor.portfolio_value = capital
    return motor


def _operaciones(motor):
    return [(t.ticker, t.side, t.quantity) for t in motor.trades]


class TestMotorPesosAOrdenes:
    def test_fracciones_y_orden_por_diferencia(self):
        # Compra fraccionaria; se opera primero la diferencia menor (BBB: 4000 < AAA: 6000) y con
        # el efectivo justo la última compra se recorta para pagar comisión (arreglo de GrowSave)
        motor = _motor()
        motor._execute_rebalance({"AAA": 0.6, "BBB": 0.4}, {"AAA": 100.0, "BBB": 50.0}, DIA)
        assert [(t.ticker, t.side) for t in motor.trades] == [("BBB", "buy"), ("AAA", "buy")]
        assert motor.trades[0].quantity == pytest.approx(79.92007992007993, rel=1e-12)
        assert motor.trades[0].commission == pytest.approx(4.0, rel=1e-12)
        assert motor.trades[1].quantity == pytest.approx(59.84025964045945, rel=1e-12)
        assert motor.cash == pytest.approx(0.0, abs=1e-9)
        assert motor.rotaciones == [(DIA, pytest.approx(1.0))]

    def test_ventas_antes_que_compras_y_rotacion_total(self):
        motor = _motor(0.0, 0.0, posiciones={"AAA": (100.0, 100.0)}, efectivo=0.0)
        motor._execute_rebalance({"BBB": 1.0}, {"AAA": 100.0, "BBB": 50.0}, DIA)
        assert _operaciones(motor) == [("AAA", "sell", 100.0), ("BBB", "buy", 200.0)]
        assert motor.rotaciones[0][1] == pytest.approx(1.0)

    def test_umbral_de_10_unidades_de_divisa(self):
        # AAA 0.5005 -> diferencia de 5 USD (< 10): no se opera nada; con 0.52 sí
        motor = _motor(0.0, 0.0, posiciones={"AAA": (50.0, 100.0), "BBB": (100.0, 50.0)}, efectivo=0.0)
        motor._execute_rebalance({"AAA": 0.5005, "BBB": 0.4995}, {"AAA": 100.0, "BBB": 50.0}, DIA)
        assert motor.trades == []
        assert motor.rotaciones[0][1] == pytest.approx(0.0005, abs=1e-9)
        motor._execute_rebalance({"AAA": 0.52, "BBB": 0.48}, {"AAA": 100.0, "BBB": 50.0}, DIA)
        assert _operaciones(motor) == [("BBB", "sell", pytest.approx(4.0)), ("AAA", "buy", pytest.approx(2.0))]

    def test_suma_distinta_de_uno_se_normaliza_solo_si_se_desvia_mas_de_0_01(self):
        # Suma 0.6: se normaliza a 1 (queda invertido todo)
        motor = _motor(0.0, 0.0)
        motor._execute_rebalance({"AAA": 0.3, "BBB": 0.3}, {"AAA": 100.0, "BBB": 50.0}, DIA)
        assert motor.cash == pytest.approx(0.0, abs=1e-9)
        # Suma 0.995 (dentro de la tolerancia): no se normaliza, 50 USD quedan en caja
        motor = _motor(0.0, 0.0)
        motor._execute_rebalance({"AAA": 0.5, "BBB": 0.495}, {"AAA": 100.0, "BBB": 50.0}, DIA)
        assert motor.cash == pytest.approx(50.0)
        assert motor.rotaciones[0][1] == pytest.approx(0.995)

    def test_cash_parcial_no_genera_orden_y_su_rotacion_cuenta_doble(self):
        # CASH no tiene precio: se salta. La rotación actual lo cuenta como valor operado (1.0 en vez
        # del 0.5 real); se fija tal cual porque GrowSave/experimentos guardados dependen de ella
        motor = _motor()
        motor._execute_rebalance({"AAA": 0.5, "CASH": 0.5}, {"AAA": 100.0}, DIA)
        assert [t.ticker for t in motor.trades] == ["AAA"]
        assert motor.cash == pytest.approx(4995.0)
        assert motor.rotaciones[0][1] == pytest.approx(1.0)

    def test_ticker_sin_precio_no_se_compra_ni_se_cierra(self):
        # AAA compra; ZZZ (sin precio hoy) no se puede cerrar y queda en cartera; la rotación lo valora a 0
        motor = _motor(0.0, 0.0, posiciones={"ZZZ": (10.0, 1.0)}, efectivo=9000.0)
        motor._execute_rebalance({"AAA": 1.0}, {"AAA": 100.0}, DIA)
        assert set(motor.positions) == {"ZZZ", "AAA"} and motor.positions["ZZZ"]["quantity"] == 10.0
        assert motor.positions["AAA"]["quantity"] == pytest.approx(90.0)
        assert motor.rotaciones[0][1] == pytest.approx(0.95)

    def test_posicion_fuera_del_objetivo_o_con_peso_cero_se_cierra_primero(self):
        motor = _motor(0.0, 0.0, posiciones={"AAA": (10.0, 100.0), "BBB": (20.0, 50.0)}, efectivo=8000.0)
        motor._execute_rebalance({"AAA": 0.0, "CCC": 1.0}, {"AAA": 100.0, "BBB": 50.0, "CCC": 10.0}, DIA)
        assert [(t.ticker, t.side) for t in motor.trades[:2]] == [("AAA", "sell"), ("BBB", "sell")]
        assert set(motor.positions) == {"CCC"}


def _backtest_con_cash():
    """Backtest mensual sobre 2 activos sintéticos con el 50 % en CASH: equity y rotaciones completas."""
    fechas = pd.bdate_range("2022-06-01", "2023-12-29")
    datos = {
        "AAA": pd.DataFrame({"close": 100 * 1.0004 ** pd.RangeIndex(len(fechas))}, index=fechas),
        "BBB": pd.DataFrame({"close": 50 * 1.0002 ** pd.RangeIndex(len(fechas))}, index=fechas),
    }
    cfg = BacktestConfig(start_date="2023-01-02", end_date="2023-12-29", initial_capital=10_000.0,
                         rebalance_frequency="monthly", commission_rate=0.001, slippage_pct=0.001)
    motor = BacktestEngine(cfg)
    resultado = motor.run_backtest(lambda fecha, d: {"AAA": 0.3, "BBB": 0.2, "CASH": 0.5}, datos)
    return motor, resultado


def test_backtest_completo_con_cash_equity_y_rotaciones_fijadas():
    motor, resultado = _backtest_con_cash()
    assert len(resultado.rotaciones) == 12
    assert all(t.ticker != "CASH" for t in motor.trades)
    curva = pd.DataFrame(motor.equity_curve)
    assert curva["equity"].iloc[-1] == pytest.approx(PIN_EQUITY_FINAL, abs=1e-6)
    assert curva["cash"].iloc[-1] == pytest.approx(PIN_CAJA_FINAL, abs=1e-6)
    # Rotación fantasma del CASH: ~0.5 en cada rebalanceo aunque casi no haya deriva
    assert [round(r, 3) for _, r in resultado.rotaciones[1:4]] == pytest.approx([0.5, 0.5, 0.5], abs=0.01)


PIN_EQUITY_FINAL = 10413.350749977026
PIN_CAJA_FINAL = 5180.9859163278015


class TestBotPesosAOrdenes:
    """El Ejecutor con POLITICA_BOT (antes `decidir_ordenes_rebalanceo`: bot de Alpaca y Nautilus): enteros, umbral 1 %, bloqueos."""

    @staticmethod
    def _decidir(pesos, equity, precios, actuales, umbral_peso=0.01, tickers_bloqueados=()):
        """Misma conversión que antes de `decidir_ordenes_rebalanceo`, ahora por el Ejecutor con POLITICA_BOT."""
        politica = replace(POLITICA_BOT, umbral_relativo=umbral_peso)
        cuenta = EstadoCuenta(equity=equity, efectivo=0.0, posiciones=actuales)
        plan = planificar_rebalanceo(pesos, precios, cuenta, politica, tickers_bloqueados)
        return [(o.ticker, o.lado, int(o.cantidad), int(o.cantidad_objetivo)) for o in plan.ordenes]

    def test_acciones_enteras_con_floor_y_ventas_primero(self):
        ordenes = self._decidir({"AAA": 0.5, "BBB": 0.5}, 10_001.0, {"AAA": 33.0, "BBB": 50.0, "CCC": 10.0},
                                {"CCC": 100, "BBB": 10})
        assert ordenes == [("CCC", "venta", 100, 0), ("BBB", "compra", 90, 100), ("AAA", "compra", 151, 151)]

    def test_umbral_del_1_por_ciento_del_equity(self):
        # Cada diferencia vale exactamente el 1 % del equity: se opera con el umbral por defecto
        # y se descarta con uno algo mayor
        assert self._decidir({"AAA": 0.5, "BBB": 0.5}, 10_000.0, {"AAA": 100.0, "BBB": 100.0},
                             {"AAA": 49, "BBB": 49}, umbral_peso=0.0100001) == []
        assert self._decidir({"AAA": 0.5, "BBB": 0.5}, 10_000.0, {"AAA": 100.0, "BBB": 100.0},
                             {"AAA": 49, "BBB": 49}) == [("AAA", "compra", 1, 50), ("BBB", "compra", 1, 50)]

    def test_cierra_posicion_a_cero_aunque_este_bajo_el_umbral(self):
        assert self._decidir({}, 10_000.0, {"BBB": 1.0}, {"BBB": 3}) == [("BBB", "venta", 3, 0)]

    def test_cash_y_tickers_sin_precio_no_generan_orden(self):
        ordenes = self._decidir({"AAA": 0.5, "CASH": 0.3, "ZZZ": 0.2}, 10_000.0, {"AAA": 100.0}, {})
        assert ordenes == [("AAA", "compra", 50, 50)]

    def test_no_normaliza_la_suma_de_pesos(self):
        # Suma 0.5: se queda en efectivo la otra mitad; suma 1.5: pide más del equity (sin normalizar)
        assert self._decidir({"AAA": 0.5}, 10_000.0, {"AAA": 100.0}, {}) == [("AAA", "compra", 50, 50)]
        assert self._decidir({"AAA": 1.5}, 10_000.0, {"AAA": 100.0}, {}) == [("AAA", "compra", 150, 150)]

    def test_peso_negativo_se_trata_como_cero(self):
        assert self._decidir({"AAA": -0.4}, 10_000.0, {"AAA": 100.0}, {"AAA": 10}) == [("AAA", "venta", 10, 0)]

    def test_bloqueados_no_se_tocan_ni_para_cerrar(self):
        ordenes = self._decidir({"AAA": 1.0}, 10_000.0, {"AAA": 100.0, "BBB": 10.0}, {"BBB": 5},
                                tickers_bloqueados={"BBB"})
        assert ordenes == [("AAA", "compra", 100, 100)]
