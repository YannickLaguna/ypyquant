"""
alpaca_vivo.py — Bucle de paper trading contra Alpaca (cuenta paper).

Responsabilidad:
    Cada N segundos: descarga precios, genera señales con ypyquant, compara con las
    posiciones de Alpaca, decide órdenes (`decidir_orden`) y las envía como órdenes de
    mercado DAY solo con el mercado abierto. Registra los fills en JSONL.

Se relaciona con:
    - config.py: claves y parámetros.
    - precios.py: datos de entrada.
    - ypyquant.execution.nautilus: constructor_senales, ordenes, registro_fills.
    - ypyquant.estrategias: modo registro (`estrategia_registro`): pesos de una Estrategia -> rebalanceo.
    - alpaca-py (extra `trading`): TradingClient y StockHistoricalDataClient.
"""

import logging
import time
from datetime import UTC, datetime

from pipelines.paper_trading.config import ConfiguracionEstrategia, ConfiguracionServicio
from pipelines.paper_trading.precios import descargar_precios
from ypyquant.execution.nautilus.constructor_senales import construir_combinador
from ypyquant.execution.nautilus.ordenes import decidir_orden, decidir_ordenes_rebalanceo
from ypyquant.execution.nautilus.registro_fills import RegistroFills

logger = logging.getLogger(__name__)


class ControlParada:
    """Bandera compartida que SIGINT/SIGTERM ponen a False para detener el bucle limpiamente."""

    def __init__(self):
        self.en_marcha = True

    def detener(self, *_args):
        logger.info("Señal de parada recibida; deteniendo el bucle...")
        self.en_marcha = False

    def dormir(self, segundos: int):
        """Duerme segundo a segundo para poder interrumpirse."""
        for _ in range(segundos):
            if not self.en_marcha:
                break
            time.sleep(1)


def mercado_abierto(cliente) -> bool:
    """True si Alpaca informa de mercado abierto; ante error, False (no se opera)."""
    try:
        return bool(cliente.get_clock().is_open)
    except Exception as e:
        logger.warning("No se pudo comprobar el estado del mercado: %s", e)
        return False


def evaluar_y_enviar(cliente, cliente_datos, ticker, senal, estrategia, equity, posiciones, abierto):
    """Obtiene el precio actual, decide y envía la orden a Alpaca. Devuelve el order o None."""
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if not abierto:
        return None
    try:
        cotizacion = cliente_datos.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))
        precio = float(cotizacion[ticker].ask_price or cotizacion[ticker].bid_price or 0)
        cantidad_actual = int(float(posiciones[ticker].qty)) if ticker in posiciones else 0

        decision = decidir_orden(
            senal=senal,
            umbral=estrategia.umbral_senal,
            equity=equity,
            max_pos_pct=estrategia.config_riesgo.get("max_position_pct", 0.10),
            precio=precio,
            cantidad_actual=cantidad_actual,
        )
        if decision.lado is None:
            logger.debug("Sin acción %s | señal=%.3f | actual=%d | objetivo=%d",
                         ticker, senal, cantidad_actual, decision.cantidad_objetivo)
            return None

        lado = OrderSide.BUY if decision.lado == "compra" else OrderSide.SELL
        orden = cliente.submit_order(MarketOrderRequest(
            symbol=ticker, qty=decision.cantidad, side=lado, time_in_force=TimeInForce.DAY,
        ))
        logger.info("ORDEN %s: %d %s | señal=%.3f | precio≈%.2f",
                    decision.lado.upper(), decision.cantidad, ticker, senal, precio)
        return orden
    except Exception as e:
        logger.warning("No se pudo evaluar/enviar la orden de %s: %s", ticker, e)
        return None


def orden_a_dict(orden) -> dict:
    """Serializa un order de alpaca-py al dict que entiende `RegistroFills`."""
    return {
        "id": str(orden.id),
        "symbol": orden.symbol,
        "side": orden.side.value if hasattr(orden.side, "value") else str(orden.side),
        "filled_qty": str(orden.filled_qty or 0),
        "filled_avg_price": str(orden.filled_avg_price or 0),
        "filled_at": orden.filled_at.isoformat() if orden.filled_at else None,
    }


def precios_a_ohlcv(precios) -> dict:
    """DataFrame fecha x ticker de cierres -> `{ticker: OHLCV mínimo}` que esperan las estrategias del registro."""
    import pandas as pd

    return {t: pd.DataFrame({"Close": precios[t], "adjusted_close": precios[t]}).dropna() for t in precios.columns}


def rebalancear_con_estrategia_registro(cliente, estrategia_registrada, precios, abierto, registro):
    """
    Una iteración en modo registro: pide pesos a la estrategia y envía las órdenes de rebalanceo.

    `estrategia_registrada` es una instancia de `ypyquant.estrategias.Estrategia` (mantiene su `estado`
    entre iteraciones). Devuelve los pesos calculados.
    """
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    datos = precios_a_ohlcv(precios)
    pesos = estrategia_registrada.como_funcion()(precios.index[-1].to_pydatetime(), datos)
    logger.info("Pesos objetivo de %s: %s", estrategia_registrada.nombre, {t: round(w, 4) for t, w in pesos.items()})
    if not abierto:
        logger.info("Mercado cerrado: no se enviarán órdenes")
        return pesos
    equity = float(cliente.get_account().equity)
    cantidades = {p.symbol: int(float(p.qty)) for p in cliente.get_all_positions()}
    ultimos = {t: float(precios[t].dropna().iloc[-1]) for t in precios.columns if precios[t].notna().any()}
    for ticker, decision in decidir_ordenes_rebalanceo(pesos, equity, ultimos, cantidades):
        try:
            lado = OrderSide.BUY if decision.lado == "compra" else OrderSide.SELL
            orden = cliente.submit_order(MarketOrderRequest(
                symbol=ticker, qty=decision.cantidad, side=lado, time_in_force=TimeInForce.DAY))
            logger.info("ORDEN %s: %d %s (objetivo %d)", decision.lado.upper(), decision.cantidad, ticker,
                        decision.cantidad_objetivo)
            registro.registrar(orden_a_dict(orden))
        except Exception as e:  # noqa: BLE001 - una orden fallida no aborta el resto
            logger.warning("No se pudo enviar la orden de %s: %s", ticker, e)
    return pesos


def ejecutar_vivo(servicio: ConfiguracionServicio, estrategia: ConfiguracionEstrategia,
                  control: ControlParada, max_iteraciones: int | None = None):
    """Bucle principal de paper trading. `max_iteraciones` permite una pasada única (--una-vez)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    alpaca = servicio.alpaca
    cliente = TradingClient(api_key=alpaca.clave_api, secret_key=alpaca.clave_secreta, paper=alpaca.paper)
    cliente_datos = StockHistoricalDataClient(api_key=alpaca.clave_api, secret_key=alpaca.clave_secreta)
    registro = RegistroFills(servicio.ruta_fills)

    cuenta = cliente.get_account()
    logger.info("Cuenta Alpaca %s | efectivo: %s | equity: %s",
                cuenta.account_number, f"{float(cuenta.cash):,.2f}", f"{float(cuenta.equity):,.2f}")

    estrategia_registrada = None
    combinador = None
    if estrategia.estrategia_registro:
        from ypyquant.estrategias import obtener_estrategia

        estrategia_registrada = obtener_estrategia(estrategia.estrategia_registro)(estrategia.parametros_registro)
        logger.info("Modo registro: estrategia %s con parámetros %s",
                    estrategia.estrategia_registro, estrategia.parametros_registro)
    else:
        combinador = construir_combinador(estrategia.config_senales)
    logger.info("Estrategia %s | tickers %s | umbral %.2f | cada %ds",
                estrategia.nombre, estrategia.tickers, estrategia.umbral_senal, servicio.segundos_entre_iteraciones)

    iteracion = 0
    while control.en_marcha and (max_iteraciones is None or iteracion < max_iteraciones):
        iteracion += 1
        logger.info("--- Iteración %d [%s] ---", iteracion, datetime.now(UTC).strftime("%H:%M:%S UTC"))
        try:
            precios = descargar_precios(estrategia.tickers, servicio.dias_historico)
            if estrategia_registrada is not None:
                if precios.empty:
                    logger.warning("Sin precios; se reintenta en la siguiente iteración")
                else:
                    rebalancear_con_estrategia_registro(
                        cliente, estrategia_registrada, precios, mercado_abierto(cliente), registro)
            else:
                senales = combinador.generate(precios) if not precios.empty else None
                if senales is None or senales.empty:
                    logger.warning("Sin precios o sin señales; se reintenta en la siguiente iteración")
                else:
                    ultimas = senales.iloc[-1]
                    logger.info("Señales actuales: %s", ultimas.to_dict())
                    equity = float(cliente.get_account().equity)
                    posiciones = {p.symbol: p for p in cliente.get_all_positions()}
                    abierto = mercado_abierto(cliente)
                    if not abierto:
                        logger.info("Mercado cerrado: no se enviarán órdenes")
                    for ticker in estrategia.tickers:
                        if ticker not in ultimas.index:
                            continue
                        orden = evaluar_y_enviar(cliente, cliente_datos, ticker, float(ultimas[ticker]),
                                                 estrategia, equity, posiciones, abierto)
                        if orden is not None:
                            registro.registrar(orden_a_dict(orden))
        except Exception as e:
            logger.error("Error en la iteración %d: %s", iteracion, e, exc_info=True)

        if control.en_marcha and (max_iteraciones is None or iteracion < max_iteraciones):
            control.dormir(servicio.segundos_entre_iteraciones)

    logger.info("Servicio detenido limpiamente")
