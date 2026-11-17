"""
pipelines.paper_trading — CLI de paper trading (datos -> señales -> órdenes en Alpaca paper).

Responsabilidad:
    Punto de entrada y configuración del paper trading personal. Se lanza con
    `uv run python -m pipelines.paper_trading ...` (ver README, sección «Paper trading»).

Módulos:
    - config.py: configuración desde variables de entorno / JSON (claves de Alpaca, estrategia).
    - precios.py: descarga de precios históricos (yfinance).
    - alpaca_vivo.py: bucle de paper trading contra Alpaca.
    - backtest.py: backtest rápido de las señales con el motor de ypyquant.
    - __main__.py: argumentos de la CLI.

Se relaciona con:
    - ypyquant.execution.nautilus: librería (combinador de señales, decisión de órdenes, registro de fills).
    - ypyquant.portfolio.backtest_engine: motor de backtest usado por `backtest.py`.
    - Extra opcional `trading` de pyproject.toml (alpaca-py, nautilus_trader, python-dotenv, yfinance).
"""
