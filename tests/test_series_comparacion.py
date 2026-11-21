"""
test_series_comparacion.py — Tests de las funciones puras de series temporales de la comparación (drawdown,
Sharpe y volatilidad móviles a 12 meses, turnover por rebalanceo y periodicidad inferida).

Se relaciona con: src/ypyquant/estrategias/series_comparacion.py y tests/test_api_comparar_series.py (endpoint).
"""

import numpy as np
import pandas as pd
import pytest

from ypyquant.estrategias.series_comparacion import (
    drawdown_serie,
    periodos_por_anio,
    sharpe_movil,
    turnover_por_rebalanceo,
    volatilidad_movil,
)


def test_periodos_por_anio_distingue_diaria_semanal_y_mensual():
    assert periodos_por_anio(pd.bdate_range("2020-01-01", periods=60)) == 252
    assert periodos_por_anio(pd.date_range("2020-01-03", periods=30, freq="7D")) == 52
    assert periodos_por_anio(pd.date_range("2020-01-31", periods=24, freq="ME")) == 12
    assert periodos_por_anio(pd.date_range("2020-03-31", periods=8, freq="QE")) == 4
    assert periodos_por_anio(pd.DatetimeIndex(["2020-01-01"])) == 252  # sin información: diaria


def test_drawdown_es_cero_en_maximos_y_negativo_bajo_ellos():
    assert drawdown_serie([100, 110, 99, 120, 90]) == pytest.approx([0.0, 0.0, -0.1, 0.0, -0.25])
    assert drawdown_serie([100, 101, 102]) == [0.0, 0.0, 0.0]
    assert drawdown_serie([]) == []


def test_volatilidad_movil_mensual_usa_ventana_de_12_observaciones():
    rng = np.random.default_rng(1)
    retornos = rng.normal(0.01, 0.03, 40)
    equity = 100 * np.cumprod(1 + retornos)
    resultado = volatilidad_movil(equity, 12)
    assert len(resultado) == 40
    assert all(v is None for v in resultado[:12])  # las 12 primeras rentabilidades aún no completan la ventana
    esperado = np.std(retornos[1:13], ddof=1) * np.sqrt(12)  # retornos de las observaciones 1..12
    assert resultado[12] == pytest.approx(esperado)


def test_sharpe_movil_es_retorno_anual_entre_volatilidad_anual():
    rng = np.random.default_rng(2)
    retornos = rng.normal(0.01, 0.03, 30)
    equity = 100 * np.cumprod(1 + retornos)
    resultado = sharpe_movil(equity, 12)
    ventana = retornos[1:13]
    esperado = (np.prod(1 + ventana) - 1) / (np.std(ventana, ddof=1) * np.sqrt(12))
    assert resultado[12] == pytest.approx(esperado)
    assert all(v is None for v in resultado[:12])


def test_series_cortas_o_planas_no_dan_valores_falsos():
    assert sharpe_movil([100, 101, 102], 12) == [None, None, None]
    plana = [100.0] * 30
    assert sharpe_movil(plana, 12)[-1] is None  # volatilidad cero: Sharpe indefinido, no 0 ni infinito
    assert volatilidad_movil(plana, 12)[-1] == 0.0


def test_turnover_por_rebalanceo_es_media_suma_de_cambios_absolutos():
    pesos = pd.DataFrame({
        "fecha": pd.to_datetime(["2020-01-31"] * 2 + ["2020-02-28"] * 2 + ["2020-03-31"] * 3),
        "ticker": ["A", "B", "A", "B", "A", "B", "C"],
        "peso": [0.5, 0.5, 0.7, 0.3, 0.0, 0.5, 0.5],
    })
    fechas, valores = turnover_por_rebalanceo(pesos)
    assert fechas == ["2020-02-28", "2020-03-31"]  # el primer rebalanceo es la compra inicial: no cuenta
    assert valores == pytest.approx([0.2, 0.5 * (0.7 + 0.2 + 0.5)])


def test_turnover_sin_pesos_es_vacio():
    assert turnover_por_rebalanceo(pd.DataFrame(columns=["fecha", "ticker", "peso"])) == ([], [])
