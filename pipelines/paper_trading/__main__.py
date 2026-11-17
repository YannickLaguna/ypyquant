"""
__main__.py — CLI del paper trading.

Uso (desde la raíz del repo, con el extra `trading` instalado):
    uv sync --extra trading
    uv run python -m pipelines.paper_trading --backtest
    uv run python -m pipelines.paper_trading --una-vez
    uv run python -m pipelines.paper_trading --config pipelines/paper_trading/estrategia_ejemplo.json

Se relaciona con: config.py, alpaca_vivo.py, backtest.py (este fichero solo parsea argumentos y orquesta).
"""

import argparse
import logging
import signal
from pathlib import Path

from pipelines.paper_trading.config import (
    ConfiguracionEstrategia,
    ConfiguracionServicio,
    cargar_env,
)

FORMATO_LOG = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipelines.paper_trading",
                                     description="Paper trading personal con ypyquant + Alpaca (cuenta paper)")
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON con la estrategia (ver estrategia_ejemplo.json); por defecto, técnica sobre 5 tickers")
    parser.add_argument("--backtest", action="store_true",
                        help="Backtest de las señales con el motor de ypyquant (no necesita claves de Alpaca)")
    parser.add_argument("--una-vez", action="store_true", help="Una sola iteración en vivo y salir")
    parser.add_argument("--intervalo", type=int, default=None, help="Segundos entre iteraciones (sobrescribe el entorno)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = crear_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=FORMATO_LOG, datefmt="%Y-%m-%d %H:%M:%S")
    for ruidosa in ("yfinance", "urllib3", "httpx"):
        logging.getLogger(ruidosa).setLevel(logging.WARNING)
    cargar_env()

    estrategia = (ConfiguracionEstrategia.desde_json(args.config) if args.config
                  else ConfiguracionEstrategia())

    if args.backtest:
        from pipelines.paper_trading.backtest import ejecutar_backtest
        ejecutar_backtest(estrategia)
        return

    from pipelines.paper_trading.alpaca_vivo import ControlParada, ejecutar_vivo
    servicio = ConfiguracionServicio.desde_entorno()
    if args.intervalo:
        servicio.segundos_entre_iteraciones = args.intervalo
    control = ControlParada()
    signal.signal(signal.SIGINT, control.detener)
    signal.signal(signal.SIGTERM, control.detener)
    ejecutar_vivo(servicio, estrategia, control, max_iteraciones=1 if args.una_vez else None)


if __name__ == "__main__":
    main()
