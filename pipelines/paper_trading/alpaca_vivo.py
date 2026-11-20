"""
alpaca_vivo.py — Bucle de paper trading contra Alpaca (cuenta paper).

Responsabilidad:
    Cada N segundos: descarga precios, genera señales con ypyquant, compara con las
    posiciones de Alpaca, decide órdenes (`decidir_orden`) y las envía como órdenes de
    mercado DAY solo con el mercado abierto.

    Modo registro (`rebalancear_con_estrategia_registro`), reglas del bucle:
    - Historia: se descarga la que pide la estrategia (`Estrategia.sesiones_necesarias`), con el env
      como mínimo; si aun así falta, WARNING claro (la estrategia usa su caso de reserva y opera igual).
    - Frecuencia: solo se rebalancea cuando toca (diaria/semanal/mensual/trimestral, la de la estrategia
      o la indicada); la fecha del último rebalanceo se persiste (estado_bot.py) y sobrevive a reinicios.
      Los demás días solo se loguea.
    - Órdenes abiertas: no se envía orden nueva sobre un ticker con una orden aún abierta, y si no se
      pueden consultar las abiertas no se envía nada (evita duplicar).
    - Fills: al enviar una orden su `filled_qty` es 0, así que solo se anota su id como pendiente;
      `sincronizar_fills` consulta después cada pendiente y registra el fill REAL (una sola vez).

Se relaciona con:
    - config.py: claves y parámetros.
    - precios.py: datos de entrada y cuánta historia pedir; estado_bot.py: frecuencia y órdenes pendientes.
    - ypyquant.execution.nautilus: constructor_senales, ordenes, registro_fills.
    - ypyquant.estrategias: modo registro (`estrategia_registro`): pesos de una Estrategia -> rebalanceo.
    - alpaca-py (extra `trading`): TradingClient y StockHistoricalDataClient.
"""

import json
import logging
import time
from datetime import UTC, date, datetime

from pipelines.paper_trading.config import ConfiguracionEstrategia, ConfiguracionServicio
from pipelines.paper_trading.estado_bot import EstadoBot, toca_rebalancear
from pipelines.paper_trading.precios import (
    avisar_historia_insuficiente,
    descargar_precios,
    dias_naturales_para_sesiones,
    resolver_dias_historico,
)
from ypyquant.estrategias.base import FRECUENCIAS_VALIDAS
from ypyquant.execution.nautilus.constructor_senales import construir_combinador
from ypyquant.execution.nautilus.ordenes import decidir_orden, decidir_ordenes_rebalanceo
from ypyquant.execution.nautilus.registro_fills import RegistroFills

logger = logging.getLogger(__name__)

# Estados de una orden de Alpaca tras los cuales ya no cambia (si hay `filled_qty`, ese fill es el definitivo)
ESTADOS_TERMINALES = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"}
# Al cerrar una iteración se espera a que las órdenes de mercado recién enviadas se ejecuten para registrar su fill
REINTENTOS_FILLS = 3
ESPERA_FILLS_S = 2.0


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


def tickers_con_orden_abierta(cliente) -> set[str]:
    """Tickers con alguna orden aún abierta en Alpaca (nueva, aceptada, parcial...). Propaga el error de red."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    abiertas = cliente.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
    return {orden.symbol for orden in abiertas}


def _abiertas_o_none(cliente) -> set[str] | None:
    """Tickers con orden abierta, o None (con WARNING) si no se pudo consultar: entonces no se envía nada."""
    try:
        return tickers_con_orden_abierta(cliente)
    except Exception as e:  # noqa: BLE001 - cualquier fallo del broker bloquea el envío, no el bucle
        logger.warning("No se pudieron consultar las órdenes abiertas (%s): no se envían órdenes para no duplicarlas", e)
        return None


def _valor_enum(valor) -> str:
    """`OrderStatus.FILLED` o 'filled' -> 'filled'."""
    return str(getattr(valor, "value", valor)).lower()


def sincronizar_fills(cliente, registro: RegistroFills, estado: EstadoBot,
                      reintentos: int = REINTENTOS_FILLS, espera_s: float = ESPERA_FILLS_S) -> int:
    """
    Registra el fill real de las órdenes enviadas que ya han terminado y las quita de las pendientes.

    Al enviar una orden de mercado su `filled_qty` es 0 (aún no se ha ejecutado), así que el fill no puede
    anotarse entonces: se guarda su id (`estado.ordenes_pendientes`) y aquí se consulta con `get_order_by_id`.
    Una orden en estado terminal (filled, canceled, expired, rejected...) con `filled_qty` > 0 se registra una
    sola vez (también el fill parcial de una cancelada); sin cantidad se descarta. Las aún abiertas (new,
    accepted, partially_filled) o que no se pudieron consultar siguen pendientes. Con `reintentos` > 0 espera
    `espera_s` entre consultas, para dar tiempo a que las órdenes de mercado recién enviadas se ejecuten.
    Devuelve cuántos fills se registraron.
    """
    registrados = 0
    for intento in range(reintentos + 1):
        for id_orden in list(estado.ordenes_pendientes):
            try:
                orden = cliente.get_order_by_id(id_orden)
            except Exception as e:  # noqa: BLE001 - se reintenta en la siguiente sincronización
                logger.warning("No se pudo consultar la orden %s: %s", id_orden, e)
                continue
            situacion = _valor_enum(orden.status)
            if situacion not in ESTADOS_TERMINALES:
                continue
            if float(orden.filled_qty or 0) > 0:
                if registro.registrar(orden_a_dict(orden)) is not None:
                    registrados += 1
                    logger.info("FILL registrado: %s %s %s @ %s", orden.symbol, _valor_enum(orden.side),
                                orden.filled_qty, orden.filled_avg_price)
            else:
                logger.info("Orden %s de %s terminó sin ejecutarse (%s)", id_orden, orden.symbol, situacion)
            estado.quitar_pendiente(id_orden)
        if not estado.ordenes_pendientes or intento == reintentos:
            break
        time.sleep(espera_s)
    return registrados


def _sincronizar_sin_fallar(cliente, registro: RegistroFills, estado: EstadoBot, **opciones) -> None:
    """Sincroniza fills sin que un error de red tumbe la iteración (se reintenta en la siguiente)."""
    try:
        sincronizar_fills(cliente, registro, estado, **opciones)
    except Exception as e:  # noqa: BLE001
        logger.warning("No se pudieron sincronizar los fills: %s", e)


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


def rebalancear_con_estrategia_registro(cliente, estrategia_registrada, precios, abierto, estado: EstadoBot,
                                        frecuencia: str, hoy: date | None = None):
    """
    Una iteración en modo registro: si toca, pide pesos a la estrategia y envía las órdenes de rebalanceo.

    `estrategia_registrada` es una instancia de `ypyquant.estrategias.Estrategia` (mantiene su `estado`
    entre iteraciones). Reglas:
    - Solo rebalancea cuando `toca_rebalancear(frecuencia, estado.ultimo_rebalanceo, hoy)`; si no, loguea
      y devuelve None (no calcula pesos ni envía nada).
    - Con el mercado cerrado calcula y muestra los pesos pero no envía ni marca el rebalanceo.
    - No envía orden sobre un ticker con una orden abierta; si no puede consultar las abiertas, no envía nada.
    - Cada orden enviada queda como pendiente de registrar (su fill llega después, ver `sincronizar_fills`).
    - Marca el rebalanceo como hecho solo si ningún envío falló (si falla uno, se reintenta en la siguiente
      iteración; las ya enviadas quedan protegidas por el bloqueo de órdenes abiertas).

    Devuelve los pesos calculados, o None si hoy no tocaba.
    """
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    hoy = hoy or datetime.now(UTC).date()
    if not toca_rebalancear(frecuencia, estado.ultimo_rebalanceo, hoy):
        logger.info("Hoy no toca rebalancear (frecuencia %s, último rebalanceo %s)", frecuencia, estado.ultimo_rebalanceo)
        return None

    datos = precios_a_ohlcv(precios)
    pesos = estrategia_registrada.como_funcion()(precios.index[-1].to_pydatetime(), datos)
    logger.info("Pesos objetivo de %s: %s", estrategia_registrada.nombre, {t: round(w, 4) for t, w in pesos.items()})
    if not abierto:
        logger.info("Mercado cerrado: no se enviarán órdenes")
        return pesos
    bloqueados = _abiertas_o_none(cliente)
    if bloqueados is None:
        return pesos
    equity = float(cliente.get_account().equity)
    cantidades = {p.symbol: int(float(p.qty)) for p in cliente.get_all_positions()}
    ultimos = {t: float(precios[t].dropna().iloc[-1]) for t in precios.columns if precios[t].notna().any()}
    afectados = sorted(bloqueados & (set(pesos) | set(cantidades)))
    if afectados:
        logger.info("No se envían órdenes nuevas para %s: ya hay una orden abierta sobre cada uno", ", ".join(afectados))

    hubo_error = False
    for ticker, decision in decidir_ordenes_rebalanceo(pesos, equity, ultimos, cantidades, tickers_bloqueados=bloqueados):
        try:
            lado = OrderSide.BUY if decision.lado == "compra" else OrderSide.SELL
            orden = cliente.submit_order(MarketOrderRequest(
                symbol=ticker, qty=decision.cantidad, side=lado, time_in_force=TimeInForce.DAY))
            logger.info("ORDEN %s: %d %s (objetivo %d)", decision.lado.upper(), decision.cantidad, ticker,
                        decision.cantidad_objetivo)
            estado.anadir_pendiente(str(orden.id))  # el fill real se registra en `sincronizar_fills`
        except Exception as e:  # noqa: BLE001 - una orden fallida no aborta el resto
            hubo_error = True
            logger.warning("No se pudo enviar la orden de %s: %s", ticker, e)
    if hubo_error:
        logger.warning("Algún envío falló: el rebalanceo no se da por hecho y se reintentará en la siguiente iteración")
    else:
        estado.marcar_rebalanceo(hoy)
    return pesos


def crear_clientes(servicio: ConfiguracionServicio):
    """Crea el `TradingClient` (paper) y el cliente de datos de Alpaca; aislado para poder falsearlo en los tests."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    alpaca = servicio.alpaca
    cliente = TradingClient(api_key=alpaca.clave_api, secret_key=alpaca.clave_secreta, paper=alpaca.paper)
    cliente_datos = StockHistoricalDataClient(api_key=alpaca.clave_api, secret_key=alpaca.clave_secreta)
    return cliente, cliente_datos


def _firma_estrategia(estrategia: ConfiguracionEstrategia) -> str:
    """Identifica estrategia + parámetros + tickers: el último rebalanceo guardado solo vale para esta misma firma."""
    modo = estrategia.estrategia_registro or "senales"
    parametros = json.dumps(estrategia.parametros_registro, sort_keys=True, ensure_ascii=False)
    return f"{modo}|{parametros}|{','.join(sorted(estrategia.tickers))}"


def ejecutar_vivo(servicio: ConfiguracionServicio, estrategia: ConfiguracionEstrategia,
                  control: ControlParada, max_iteraciones: int | None = None):
    """Bucle principal de paper trading. `max_iteraciones` permite una pasada única (--una-vez)."""
    cliente, cliente_datos = crear_clientes(servicio)
    registro = RegistroFills(servicio.ruta_fills)
    estado = EstadoBot.cargar(servicio.ruta_estado, _firma_estrategia(estrategia))

    cuenta = cliente.get_account()
    logger.info("Cuenta Alpaca %s | efectivo: %s | equity: %s",
                cuenta.account_number, f"{float(cuenta.cash):,.2f}", f"{float(cuenta.equity):,.2f}")

    estrategia_registrada = None
    combinador = None
    frecuencia = None
    sesiones_necesarias = 0
    dias_historico = servicio.dias_historico
    if estrategia.estrategia_registro:
        from ypyquant.estrategias import obtener_estrategia

        clase = obtener_estrategia(estrategia.estrategia_registro)
        estrategia_registrada = clase(estrategia.parametros_registro)
        frecuencia = estrategia.frecuencia_rebalanceo_registro or clase.frecuencia_rebalanceo
        if frecuencia not in FRECUENCIAS_VALIDAS:
            raise ValueError(f"Frecuencia de rebalanceo no válida: {frecuencia!r} (válidas: {', '.join(FRECUENCIAS_VALIDAS)})")
        sesiones_necesarias = estrategia_registrada.sesiones_necesarias()
        dias_historico = resolver_dias_historico(sesiones_necesarias, servicio.dias_historico)
        logger.info("Modo registro: estrategia %s con parámetros %s, rebalanceo %s",
                    estrategia.estrategia_registro, estrategia.parametros_registro, frecuencia)
        logger.info("Historia: la estrategia necesita %d sesiones (≈%d días naturales); mínimo configurado %d días -> se descargan %d días",
                    sesiones_necesarias, dias_naturales_para_sesiones(sesiones_necesarias), servicio.dias_historico, dias_historico)
        if estado.ultimo_rebalanceo:
            logger.info("Último rebalanceo registrado: %s", estado.ultimo_rebalanceo)
    else:
        combinador = construir_combinador(estrategia.config_senales)
    logger.info("Estrategia %s | tickers %s | umbral %.2f | cada %ds",
                estrategia.nombre, estrategia.tickers, estrategia.umbral_senal, servicio.segundos_entre_iteraciones)

    iteracion = 0
    while control.en_marcha and (max_iteraciones is None or iteracion < max_iteraciones):
        iteracion += 1
        hoy = datetime.now(UTC).date()
        logger.info("--- Iteración %d [%s] ---", iteracion, datetime.now(UTC).strftime("%H:%M:%S UTC"))
        _sincronizar_sin_fallar(cliente, registro, estado, reintentos=0)  # fills de órdenes de iteraciones anteriores
        try:
            if estrategia_registrada is not None:
                if not toca_rebalancear(frecuencia, estado.ultimo_rebalanceo, hoy):
                    logger.info("Hoy no toca rebalancear (frecuencia %s, último rebalanceo %s)", frecuencia,
                                estado.ultimo_rebalanceo)
                else:
                    precios = descargar_precios(estrategia.tickers, dias_historico)
                    if precios.empty:
                        logger.warning("Sin precios; se reintenta en la siguiente iteración")
                    else:
                        avisar_historia_insuficiente(precios, sesiones_necesarias, estrategia.nombre)
                        rebalancear_con_estrategia_registro(
                            cliente, estrategia_registrada, precios, mercado_abierto(cliente), estado, frecuencia, hoy=hoy)
            else:
                precios = descargar_precios(estrategia.tickers, dias_historico)
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
                    bloqueados = _abiertas_o_none(cliente) if abierto else set()
                    for ticker in estrategia.tickers:
                        if ticker not in ultimas.index or bloqueados is None:
                            continue
                        if ticker in bloqueados:
                            logger.info("No se envía orden para %s: ya hay una orden abierta", ticker)
                            continue
                        orden = evaluar_y_enviar(cliente, cliente_datos, ticker, float(ultimas[ticker]),
                                                 estrategia, equity, posiciones, abierto)
                        if orden is not None:
                            estado.anadir_pendiente(str(orden.id))
        except Exception as e:
            logger.error("Error en la iteración %d: %s", iteracion, e, exc_info=True)
        # Las órdenes de mercado recién enviadas suelen ejecutarse en segundos: se registra su fill real ahora
        _sincronizar_sin_fallar(cliente, registro, estado, reintentos=REINTENTOS_FILLS, espera_s=ESPERA_FILLS_S)

        if control.en_marcha and (max_iteraciones is None or iteracion < max_iteraciones):
            control.dormir(servicio.segundos_entre_iteraciones)

    logger.info("Servicio detenido limpiamente")
