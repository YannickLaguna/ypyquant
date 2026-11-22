"""
estrategia_ypyquant.py — Strategy de Nautilus Trader que usa las señales de ypyquant.

Responsabilidad:
    - `ConfigEstrategiaYpyquant`: configuración serializable (StrategyConfig de Nautilus).
    - `EstrategiaYpyquant`: en backtest/live de Nautilus acumula cierres por ticker, genera el
      SignalFrame con ypyquant y envía órdenes de mercado cuando |señal| supera el umbral.
      `generar_senales(precios)` también es usable fuera de Nautilus (modo CLI).

Se relaciona con:
    - constructor_senales.py: construye el SignalCombiner desde `signal_config`.
    - ordenes.py: `decidir_orden`, la decisión señal -> orden compartida con el bot de Alpaca (sin duplicar reglas).
    - ypyquant.signals: lógica real de las señales (esta clase es solo el envoltorio).
    - pipelines/paper_trading: el modo vivo usa Alpaca directamente; esta clase queda para
      un TradingNode/BacktestEngine de Nautilus.
    - Requiere el extra `trading` (nautilus_trader); sin él, este módulo no se puede importar.
"""

import logging

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from ypyquant.execution.nautilus.constructor_senales import construir_combinador
from ypyquant.execution.nautilus.ordenes import decidir_orden
from ypyquant.signals import SignalCombiner

logger = logging.getLogger(__name__)

SEDE = "ALPACA"
MIN_BARRAS = 30  # historia mínima por ticker antes de calcular señales


class ConfigEstrategiaYpyquant(StrategyConfig, frozen=True):
    """Configuración inmutable de la estrategia (campos con los mismos nombres que en GrowSave)."""
    tickers: tuple = ("AAPL", "MSFT", "GOOGL")
    signal_config: dict = {}
    risk_config: dict = {}
    signal_threshold: float = 0.3
    price_lookback_days: int = 252
    rebalance_freq: str = "daily"  # "daily" | "weekly"


class EstrategiaYpyquant(Strategy):
    """
    Envoltorio de Nautilus para las señales de ypyquant.

    on_start: se suscribe a barras diarias de cada ticker.
    on_bar: guarda el cierre en un buffer; con >= MIN_BARRAS en todos los tickers, genera
        señales y opera. on_stop: cancela órdenes y cierra posiciones.
    """

    def __init__(self, config: ConfigEstrategiaYpyquant):
        super().__init__(config)
        self.tickers = list(config.tickers)
        self.config_senales = config.signal_config
        self.config_riesgo = config.risk_config
        self.umbral = config.signal_threshold
        self.ventana = config.price_lookback_days
        self.frecuencia_rebalanceo = config.rebalance_freq

        # Buffer de cierres por ticker
        self._buffer_precios: dict[str, list[float]] = {t: [] for t in self.tickers}
        self._combinador: SignalCombiner | None = None

    @staticmethod
    def _tipo_barra(ticker: str) -> BarType:
        return BarType.from_str(f"{ticker}.{SEDE}-1-DAY-LAST-EXTERNAL")

    # --- Ciclo de vida de Nautilus ---

    def on_start(self):
        logger.info("EstrategiaYpyquant.on_start() — tickers: %s", self.tickers)
        self._combinador = construir_combinador(self.config_senales)
        for ticker in self.tickers:
            try:
                self.subscribe_bars(self._tipo_barra(ticker))
            except Exception as e:
                logger.warning("No se pudo suscribir a %s: %s", ticker, e)

    def on_bar(self, bar: Bar):
        ticker = bar.bar_type.instrument_id.symbol.value
        if ticker in self._buffer_precios:
            buffer = self._buffer_precios[ticker]
            buffer.append(float(bar.close))
            if len(buffer) > self.ventana:
                self._buffer_precios[ticker] = buffer[-self.ventana:]

        if min(len(v) for v in self._buffer_precios.values()) >= MIN_BARRAS:
            self._rebalancear()

    def on_stop(self):
        logger.info("EstrategiaYpyquant.on_stop()")
        self.cancel_all_orders()
        self.close_all_positions()

    # --- Señales y órdenes ---

    def _rebalancear(self):
        """Calcula las señales con el buffer actual y actúa sobre la última fila."""
        precios = self._construir_precios()
        if precios is None or precios.empty:
            return
        senales = self.generar_senales(precios)
        if senales is None or senales.empty:
            return
        ultima = senales.iloc[-1]
        for ticker in self.tickers:
            if ticker in ultima.index:
                self._actuar_sobre_senal(ticker, float(ultima[ticker]))

    def _actuar_sobre_senal(self, ticker: str, senal: float):
        """Convierte una señal en [-1, +1] en una orden de mercado de Nautilus."""
        try:
            id_instrumento = InstrumentId.from_str(f"{ticker}.{SEDE}")
            if self.cache.instrument(id_instrumento) is None:
                return
            cuenta = self.portfolio.account(Venue(SEDE))
            if cuenta is None:
                return

            equity = float(cuenta.balance_total().as_double())
            cantidad_actual = int(float(self.portfolio.net_position(id_instrumento)))

            barra = self.cache.bar(self._tipo_barra(ticker))
            if barra is None or float(barra.close) <= 0:
                return
            # Misma decisión que el bot de Alpaca en modo señales (antes aquí se repetía con un 0.9 mágico)
            decision = decidir_orden(
                senal=senal, umbral=self.umbral, equity=equity,
                max_pos_pct=self.config_riesgo.get("max_position_pct", 0.10),
                precio=float(barra.close), cantidad_actual=cantidad_actual,
            )
            if decision.lado == "compra":
                self._enviar(id_instrumento, OrderSide.BUY, decision.cantidad)
                logger.info("COMPRA %d %s (señal=%.2f)", decision.cantidad, ticker, senal)
            elif decision.lado == "venta":
                self._enviar(id_instrumento, OrderSide.SELL, decision.cantidad)
                logger.info("VENTA %d %s (señal=%.2f)", decision.cantidad, ticker, senal)
        except Exception as e:
            logger.error("Error actuando sobre la señal de %s: %s", ticker, e)

    def _enviar(self, id_instrumento: InstrumentId, lado: OrderSide, cantidad: int):
        orden = self.order_factory.market(
            instrument_id=id_instrumento,
            order_side=lado,
            quantity=Quantity.from_int(cantidad),
            time_in_force=TimeInForce.DAY,
        )
        self.submit_order(orden)

    # --- API pública: usable dentro y fuera de Nautilus ---

    def generar_senales(self, precios: pd.DataFrame) -> pd.DataFrame | None:
        """
        Genera el SignalFrame (fecha x ticker, valores en [-1, +1]) desde precios de cierre.
        Devuelve None si el combinador falla (se registra el error).
        """
        if self._combinador is None:
            self._combinador = construir_combinador(self.config_senales)
        try:
            return self._combinador.generate(precios)
        except Exception as e:
            logger.error("Error generando señales: %s", e)
            return None

    def _construir_precios(self) -> pd.DataFrame | None:
        """DataFrame de precios desde el buffer interno (índice entero 'date'); None si hay <2 barras."""
        largo_min = min(len(v) for v in self._buffer_precios.values())
        if largo_min < 2:
            return None
        df = pd.DataFrame({t: self._buffer_precios[t][-largo_min:] for t in self.tickers})
        df.index = pd.RangeIndex(largo_min)
        df.index.name = "date"
        return df
