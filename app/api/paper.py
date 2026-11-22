"""
Servicios de paper trading para la API: cliente Alpaca (cuenta paper) y control del bot como subproceso.

Responsabilidad:
- `ServicioAlpaca(perfil)`: lee cuenta, posiciones, órdenes e historial de equity de UNA cuenta paper con alpaca-py.
  Las claves del perfil vienen de `ALPACA_<PERFIL>_API_KEY` / `ALPACA_<PERFIL>_SECRET_KEY` (el perfil `DEFECTO` usa
  `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`; ver ypyquant.perfiles_alpaca.perfiles; siempre `paper=True`). Sin claves lanza
  `SinClavesAlpaca` con el nombre del perfil y las variables que faltan (la API responde 503). `comprobar_conexion`
  resume el estado (configurada, conectada, equity) sin exponer claves y con los errores redactados. Admite un
  `fabrica_cliente` inyectable para los tests. Importa alpaca-py una sola vez bajo cerrojo (`_importar_alpaca`) para
  evitar el bloqueo de importación entre hilos de la API.
- `leer_fills`: fills registrados por el bot en un JSONL (`data/paper_trading/<PERFIL>/fills.jsonl`; el perfil
  `DEFECTO` cae al antiguo `data/paper_trading/fills.jsonl` si no tiene el suyo), en el formato de
  `ypyquant.execution.nautilus.registro_fills` (las filas antiguas, sin `perfil`, se leen igual).
- `BotControl`: bot ÚNICO heredado (`/api/paper/bot/*`, estrategia suelta por petición). La UI multi-perfil usa
  `GestorBots` (app/api/gestor_bots.py), un bot por perfil vinculado a un experimento. El subproceso es hijo de la API:
  si la API muere en Windows, el bot puede seguir vivo.

Se relaciona con:
- pipelines/paper_trading (CLI que se lanza; acepta `--estrategia`, `--parametros`, `--tickers`).
- ypyquant.perfiles_alpaca (perfiles de claves y redacción de secretos).
- ypyquant.execution.nautilus.registro_fills (formato de los fills).
- app/api/rutas_paper.py (expone todo esto) y app/api/contexto.py (lo agrupa).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ypyquant.perfiles_alpaca.perfiles import NOMBRE_DEFECTO, claves_de, variables_de
from ypyquant.perfiles_alpaca.redaccion import redactar

RAIZ_REPO = Path(__file__).resolve().parents[2]
CARPETA_PAPER = RAIZ_REPO / "data" / "paper_trading"


class SinClavesAlpaca(RuntimeError):
    """Faltan las claves del perfil en el entorno (`ALPACA_<PERFIL>_API_KEY` / `ALPACA_<PERFIL>_SECRET_KEY`)."""


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


def _importar_alpaca() -> None:
    """
    Importa de una vez los módulos de alpaca-py que usa el servicio.

    La API atiende las peticiones en hilos paralelos; al cargar la página de Paper trading llegan
    varias a la vez y, si cada una importa por primera vez un módulo distinto de `alpaca.trading`
    (client, enums, requests se importan entre sí), Python detecta un bloqueo cruzado de importación
    (`_DeadlockError: deadlock detected by _ModuleLock`). Se llama bajo el cerrojo de `cliente()`.
    """
    import alpaca.trading.client  # noqa: F401
    import alpaca.trading.enums  # noqa: F401
    import alpaca.trading.requests  # noqa: F401


class ServicioAlpaca:
    """
    Lectura de UNA cuenta paper de Alpaca (un perfil).

    Args:
        fabrica_cliente: callable sin argumentos que devuelve un `TradingClient`-compatible (por defecto,
            crea el real con las claves del perfil). Se cachea tras la primera llamada.
        perfil: nombre del perfil (`DEFECTO` por defecto: variables `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`).
    """

    TTL_CONEXION_S = 15.0

    def __init__(self, fabrica_cliente: Callable[[], Any] | None = None, perfil: str = NOMBRE_DEFECTO):
        self.perfil = perfil
        self._fabrica = fabrica_cliente
        self._cliente: Any = None
        self._cerrojo = threading.Lock()
        self._conexion: tuple[float, dict[str, Any]] | None = None

    def claves_disponibles(self) -> bool:
        return claves_de(self.perfil) is not None

    def _cliente_real(self):
        claves = claves_de(self.perfil)
        if claves is None:
            api, secreto = variables_de(self.perfil)
            raise SinClavesAlpaca(
                f"Perfil {self.perfil} sin configurar: define {api} y {secreto} en el .env de la raíz del repo y reinicia la API.")
        from alpaca.trading.client import TradingClient

        return TradingClient(api_key=claves[0], secret_key=claves[1], paper=True)

    @property
    def disponible(self) -> bool:
        return self._fabrica is not None or self.claves_disponibles()

    def cliente(self):
        with self._cerrojo:
            if self._cliente is None:
                _importar_alpaca()
                self._cliente = self._fabrica() if self._fabrica else self._cliente_real()
            return self._cliente

    def comprobar_conexion(self, forzar: bool = False) -> dict[str, Any]:
        """
        Estado de la cuenta sin exponer claves: `{configurada, conectada, equity, error}`.

        Consulta la cuenta a Alpaca como mucho una vez cada `TTL_CONEXION_S` (la lista de perfiles se refresca a menudo).
        El error se redacta por si el mensaje de Alpaca citara una clave.
        """
        ahora = time.monotonic()
        if not forzar and self._conexion and ahora - self._conexion[0] < self.TTL_CONEXION_S:
            return self._conexion[1]
        if not self.disponible:
            resultado = {"configurada": False, "conectada": False, "equity": None, "error": None}
        else:
            try:
                resultado = {"configurada": True, "conectada": True, "equity": self.cuenta()["equity"], "error": None}
            except Exception as error:  # noqa: BLE001 - cualquier fallo de red o de Alpaca se resume
                resultado = {"configurada": True, "conectada": False, "equity": None, "error": redactar(str(error))}
        self._conexion = (ahora, resultado)
        return resultado

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
        cliente = self.cliente()  # primero: garantiza que alpaca ya está importado (ver `_importar_alpaca`)
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        ordenes = cliente.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limite))
        return [{
            "id": str(o.id), "ticker": o.symbol, "lado": _texto(o.side), "tipo": _texto(getattr(o, "order_type", None)),
            "cantidad": _numero(getattr(o, "qty", None)), "cantidad_ejecutada": _numero(getattr(o, "filled_qty", None)),
            "precio_ejecucion": _numero(getattr(o, "filled_avg_price", None)), "estado": _texto(o.status),
            "creada": _texto(getattr(o, "created_at", None)), "ejecutada": _texto(getattr(o, "filled_at", None)),
        } for o in ordenes]

    def historial_equity(self, periodo: str = "1M", marco: str = "1D") -> dict[str, Any]:
        cliente = self.cliente()  # primero: garantiza que alpaca ya está importado (ver `_importar_alpaca`)
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        h = cliente.get_portfolio_history(GetPortfolioHistoryRequest(period=periodo, timeframe=marco))
        fechas = [datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d %H:%M") for t in (h.timestamp or [])]
        return {"fechas": fechas, "equity": [_numero(v) for v in (h.equity or [])],
                "pnl": [_numero(v) for v in (h.profit_loss or [])], "base": _numero(getattr(h, "base_value", None))}


def leer_fills(ruta: Path | None = None, limite: int = 200, perfil: str | None = None,
               carpeta: Path | None = None) -> list[dict[str, Any]]:
    """
    Últimos `limite` fills del JSONL del bot (más recientes primero); vacío si no existe.

    Con `perfil` se lee `<carpeta>/<perfil>/fills.jsonl` (`carpeta` por defecto `data/paper_trading`); el perfil `DEFECTO`
    cae al fichero antiguo `<carpeta>/fills.jsonl` si no tiene el suyo. Las líneas corruptas se saltan. Las filas
    antiguas (sin `perfil`) se devuelven tal cual: los campos nuevos son opcionales.
    """
    carpeta = carpeta or CARPETA_PAPER
    if ruta is None and perfil:
        ruta = carpeta / perfil / "fills.jsonl"
        if perfil == NOMBRE_DEFECTO and not ruta.exists():
            ruta = carpeta / "fills.jsonl"
    ruta = ruta or carpeta / "fills.jsonl"
    if not ruta.exists():
        return []
    filas = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        try:
            filas.append(json.loads(linea))
        except json.JSONDecodeError:
            continue
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
