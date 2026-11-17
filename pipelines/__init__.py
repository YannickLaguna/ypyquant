"""
pipelines — Puntos de entrada ejecutables del entorno de trading (CLI).

Responsabilidad: agrupar los pipelines (datos -> estrategias -> backtest -> paper trading)
que usan la librería `ypyquant`. Cada subpaquete se lanza con `python -m pipelines.<nombre>`.

Se relaciona con: src/ypyquant (librería) y scripts/ (utilidades sueltas).
"""
