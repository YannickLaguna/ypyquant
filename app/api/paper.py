"""
Servicios de paper trading para la API: cliente Alpaca (cuenta paper) y control del bot como subproceso.

Responsabilidad:
- `ServicioAlpaca`: lee cuenta, posiciones, órdenes e historial de equity con alpaca-py. Las claves
  vienen de `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (siempre `paper=True`). Sin claves lanza
  `SinClavesAlpaca` (la API responde 503 con un mensaje claro). Admite un `fabrica_cliente`
  inyectable para los tests.
- `leer_fills`: fills registrados por el bot en `data/paper_trading/fills.jsonl`
  (`ypyquant.execution.nautilus.registro_fills`), si existe.
- `BotControl`: arranca y para `python -m pipelines.paper_trading` con una estrategia del
  registro, guarda su salida en `data/paper_trading/bot.log` y la sirve por offset de línea
  (polling). El subproceso es hijo de la API: si la API muere en Windows, el bot puede seguir vivo.

Se relaciona con:
- pipelines/paper_trading (CLI que se lanza; acepta `--estrategia`, `--parametros`, `--tickers`).
- ypyquant.execution.nautilus.registro_fills (formato de los fills).
- app/api/rutas_paper.py (expone todo esto) y app/api/contexto.py (lo agrupa).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RAIZ_REPO = Path(__file__).resolve().parents[2]
CARPETA_PAPER = RAIZ_REPO / "data" / "paper_trading"


class SinClavesAlpaca(RuntimeError):
    """No hay ALPACA_API_KEY / ALPACA_SECRET_KEY en el entorno."""


def _numero(valor: Any) -> float | None:
    """Alpaca devuelve importes como cadenas; los pasamos a float (None si no hay)."""
    try:
        return None if valor is None else float(valor)
    except (TypeError, ValueError):
        return None


def _texto(valor: Any) -> str | None:
    """Enums de Alpaca (`.value`) y fechas a texto."""
    if valor is None:
        return None
    if hasattr(valor, "isoformat"):
        return valor.isoformat()
    return str(getattr(valor, "value", valor))


class ServicioAlpaca:
    """
    Lectura de la cuenta paper de Alpaca.

    Args:
        fabrica_cliente: callable sin argumentos que devuelve un `TradingClient`-compatible (por defecto,
            crea el real con las claves del entorno). Se cachea tras la primera llamada.
    """

    def __init__(self, fabrica_cliente: Callable[[], Any] | None = None):
        self._fabrica = fabrica_cliente
        self._cliente: Any = None
        self._cerrojo = threading.Lock()

    @staticmethod
    def claves_disponibles() -> bool:
        return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))

    def _cliente_real(self):
        if not self.claves_disponibles():
            raise SinClavesAlpaca(
                "Faltan ALPACA_API_KEY y ALPACA_SECRET_KEY en el entorno (copia .env.example a .env y reinicia la API)."
            )
        from alpaca.trading.client import TradingClient

        return TradingClient(api_key=os.environ["ALPACA_API_KEY"], secret_key=os.environ["ALPACA_SECRET_KEY"], paper=True)

    @property
    def disponible(self) -> bool:
        return self._fabrica is not None or self.claves_disponibles()

    def cliente(self):
        with self._cerrojo:
            if self._cliente is None:
                self._cliente = self._fabrica() if self._fabrica else self._cliente_real()
            return self._cliente

    def cuenta(self) -> dict[str, Any]:
        c = self.cliente().get_account()
        equity, ultimo = _numero(c.equity), _numero(getattr(c, "last_equity", None))
        return {
            "numero": getattr(c, "account_number", None), "estado": _texto(getattr(c, "status", None)),
            "equity": equity, "efectivo": _numero(c.cash), "poder_compra": _numero(getattr(c, "buying_power", None)),
            "valor_cartera": _numero(getattr(c, "portfolio_value", None)), "equity_cierre_anterior": ultimo,
            "pnl_dia": None if equity is None or ultimo is None else equity - ultimo,
            "pnl_dia_pct": None if not equity or not ultimo else equity / ultimo - 1.0,
        }

    def posiciones(self) -> list[dict[str, Any]]:
        return [{
            "ticker": p.symbol, "cantidad": _numero(p.qty), "lado": _texto(getattr(p, "side", None)),
            "precio_medio": _numero(p.avg_entry_price), "precio_actual": _numero(getattr(p, "current_price", None)),
            "valor_mercado": _numero(getattr(p, "market_value", None)),
            "pnl": _numero(getattr(p, "unrealized_pl", None)), "pnl_pct": _numero(getattr(p, "unrealized_plpc", None)),
        } for p in self.cliente().get_all_positions()]

    def ordenes(self, limite: int = 100) -> list[dict[str, Any]]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        ordenes = self.cliente().get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limite))
        return [{
            "id": str(o.id), "ticker": o.symbol, "lado": _texto(o.side), "tipo": _texto(getattr(o, "order_type", None)),
            "cantidad": _numero(getattr(o, "qty", None)), "cantidad_ejecutada": _numero(getattr(o, "filled_qty", None)),
            "precio_ejecucion": _numero(getattr(o, "filled_avg_price", None)), "estado": _texto(o.status),
            "creada": _texto(getattr(o, "created_at", None)), "ejecutada": _texto(getattr(o, "filled_at", None)),
        } for o in ordenes]

    def historial_equity(self, periodo: str = "1M", marco: str = "1D") -> dict[str, Any]:
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        h = self.cliente().get_portfolio_history(GetPortfolioHistoryRequest(period=periodo, timeframe=marco))
        fechas = [datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d %H:%M") for t in (h.timestamp or [])]
        return {"fechas": fechas, "equity": [_numero(v) for v in (h.equity or [])],
                "pnl": [_numero(v) for v in (h.profit_loss or [])], "base": _numero(getattr(h, "base_value", None))}


def leer_fills(ruta: Path | None = None, limite: int = 200) -> list[dict[str, Any]]:
    """Últimos `limite` fills del JSONL del bot (más recientes primero); vacío si no existe."""
    ruta = ruta or CARPETA_PAPER / "fills.jsonl"
    if not ruta.exists():
        return []
    filas = [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]
    return filas[-limite:][::-1]


class BotControl:
    """
    Arranca y detiene el bot de paper trading como subproceso y sirve su log.

    Args:
        carpeta: donde se guarda `bot.log`.
        construir_comando: `(estrategia, parametros, tickers, intervalo_s, **extras) -> argv` (extras opcionales:
            `dias_historico`, `frecuencia`); por defecto lanza `python -m pipelines.paper_trading --estrategia ...`
            (se inyecta otro en los tests).
    """

    def __init__(self, carpeta: Path | None = None, construir_comando: Callable[..., list[str]] | None = None):
        self.carpeta = Path(carpeta) if carpeta else CARPETA_PAPER
        self.construir_comando = construir_comando or self._comando_por_defecto
        self._proceso: subprocess.Popen | None = None
        self._meta: dict[str, Any] = {}
        self._cerrojo = threading.Lock()

    @staticmethod
    def _comando_por_defecto(estrategia: str, parametros: dict, tickers: list[str], intervalo_s: int,
                             dias_historico: int | None = None, frecuencia: str | None = None) -> list[str]:
        """Comando del CLI; `--dias-historico` y `--frecuencia` solo van si se pidieron (si no, decide la estrategia)."""
        comando = [sys.executable, "-m", "pipelines.paper_trading", "--estrategia", estrategia,
                   "--parametros", json.dumps(parametros), "--tickers", ",".join(tickers), "--intervalo", str(intervalo_s)]
        if dias_historico is not None:
            comando += ["--dias-historico", str(dias_historico)]
        if frecuencia is not None:
            comando += ["--frecuencia", frecuencia]
        return comando

    @property
    def ruta_log(self) -> Path:
        return self.carpeta / "bot.log"

    def en_marcha(self) -> bool:
        return self._proceso is not None and self._proceso.poll() is None

    def estado(self) -> dict[str, Any]:
        codigo = None if self._proceso is None else self._proceso.poll()
        return {"en_marcha": self.en_marcha(), "pid": self._proceso.pid if self._proceso else None,
                "codigo_salida": codigo, **self._meta}

    def arrancar(self, estrategia: str, parametros: dict, tickers: list[str], intervalo_s: int = 60,
                 dias_historico: int | None = None, frecuencia: str | None = None) -> dict[str, Any]:
        """
        Lanza el bot; `RuntimeError` si ya hay uno en marcha.

        `dias_historico` (mínimo de días naturales a descargar) y `frecuencia` (de rebalanceo) son opcionales:
        sin ellos el bot usa lo que necesite la estrategia y su frecuencia sugerida. Solo se pasan a
        `construir_comando` cuando se indican.
        """
        extras = {k: v for k, v in (("dias_historico", dias_historico), ("frecuencia", frecuencia)) if v is not None}
        with self._cerrojo:
            if self.en_marcha():
                raise RuntimeError("El bot ya está en marcha; páralo antes de lanzar otro")
            self.carpeta.mkdir(parents=True, exist_ok=True)
            entorno = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            with self.ruta_log.open("wb") as salida:
                self._proceso = subprocess.Popen(
                    self.construir_comando(estrategia, parametros, tickers, intervalo_s, **extras),
                    cwd=RAIZ_REPO, stdout=salida, stderr=subprocess.STDOUT, env=entorno,
                )
            self._meta = {"estrategia": estrategia, "parametros": parametros, "tickers": tickers,
                          "intervalo_s": intervalo_s, "inicio": datetime.now(UTC).isoformat()}
            if dias_historico is not None:
                self._meta["dias_historico"] = dias_historico
            if frecuencia is not None:
                self._meta["frecuencia_rebalanceo"] = frecuencia
        return self.estado()

    def parar(self, espera_s: float = 10.0) -> dict[str, Any]:
        """Termina el subproceso (SIGTERM; en Windows terminate) y espera; lo mata si no responde."""
        with self._cerrojo:
            if self.en_marcha():
                self._proceso.terminate()
                try:
                    self._proceso.wait(timeout=espera_s)
                except subprocess.TimeoutExpired:
                    self._proceso.kill()
                    self._proceso.wait()
        return self.estado()

    def log(self, desde: int = 0, maximo: int = 500) -> dict[str, Any]:
        """Líneas del log desde la línea `desde` (offset para polling) y el siguiente offset."""
        if not self.ruta_log.exists():
            return {"lineas": [], "siguiente": 0}
        lineas = self.ruta_log.read_text(encoding="utf-8", errors="replace").splitlines()
        trozo = lineas[desde:desde + maximo]
        return {"lineas": trozo, "siguiente": desde + len(trozo)}
