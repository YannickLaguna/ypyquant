"""
ypyquant.execution.nautilus — integración opcional con Nautilus Trader.

Responsabilidad:
    Reunir la parte reutilizable (librería) de la ejecución con Nautilus/Alpaca:
      - constructor_senales: construye el SignalCombiner desde un dict de config (sin Nautilus).
      - ordenes: decisión pura señal -> orden (sin Nautilus ni red).
      - registro_fills: persistencia local (JSONL) de fills de Alpaca, sin base de datos.
      - estrategia_ypyquant: Strategy de Nautilus que envuelve las señales de ypyquant.

Se relaciona con:
    - ypyquant.signals: generadores y combinador de señales.
    - pipelines/paper_trading: CLI que orquesta datos -> señales -> órdenes en Alpaca paper.
    - Extra opcional `trading` de pyproject.toml (nautilus_trader, alpaca-py): solo
      `estrategia_ypyquant` lo exige; el resto de módulos funciona sin él.

No se importa nada pesado aquí para que `import ypyquant.execution.nautilus` no exija el extra.
"""
