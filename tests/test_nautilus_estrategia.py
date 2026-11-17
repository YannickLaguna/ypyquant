"""
test_nautilus_estrategia.py — Tests de la Strategy de Nautilus (exigen el extra `trading`).

Responsabilidad:
    Portar los tests de configuración de `YpyquantStrategyConfig` de GrowSave y comprobar que la
    estrategia genera señales con datos sintéticos. Sin red ni claves. Si `nautilus_trader` no
    está instalado, el módulo entero se salta (`importorskip`) en vez de fallar.

Se relaciona con:
    - ypyquant.execution.nautilus.estrategia_ypyquant (código bajo prueba).
    - test_paper_trading.py: el resto de tests, que no necesitan el extra.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nautilus_trader", reason="requiere el extra `trading` (uv sync --extra trading)")

from ypyquant.execution.nautilus.estrategia_ypyquant import (
    ConfigEstrategiaYpyquant,
    EstrategiaYpyquant,
)


def test_config_por_defecto():
    cfg = ConfigEstrategiaYpyquant()
    assert isinstance(cfg.tickers, tuple) and len(cfg.tickers) > 0
    assert isinstance(cfg.signal_threshold, float)


def test_config_personalizada():
    cfg = ConfigEstrategiaYpyquant(tickers=("AAPL", "MSFT"), signal_threshold=0.5,
                                   risk_config={"max_position_pct": 0.05})
    assert cfg.tickers == ("AAPL", "MSFT")
    assert cfg.signal_threshold == 0.5
    assert cfg.risk_config["max_position_pct"] == 0.05


def test_estrategia_genera_senales_sin_nautilus_en_marcha():
    estrategia = EstrategiaYpyquant(ConfigEstrategiaYpyquant(tickers=("AAPL", "MSFT")))
    rng = np.random.default_rng(1)
    precios = pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 80))) for t in ("AAPL", "MSFT")},
        index=pd.date_range("2023-01-01", periods=80, freq="B"),
    )
    senales = estrategia.generar_senales(precios)
    assert senales is not None and (senales.abs() <= 1.0).all().all()


def test_buffer_a_dataframe():
    estrategia = EstrategiaYpyquant(ConfigEstrategiaYpyquant(tickers=("AAPL", "MSFT")))
    assert estrategia._construir_precios() is None  # sin barras todavía
    estrategia._buffer_precios = {"AAPL": [1.0, 2.0, 3.0], "MSFT": [4.0, 5.0]}
    df = estrategia._construir_precios()
    assert df.shape == (2, 2) and list(df.columns) == ["AAPL", "MSFT"]
