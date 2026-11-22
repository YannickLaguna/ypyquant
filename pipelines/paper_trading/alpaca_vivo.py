"""
alpaca_vivo.py — Bucle de paper trading contra Alpaca (cuenta paper).

Responsabilidad:
    Orquestación fina del bucle: calendario -> estrategia -> Ejecutor -> broker. Cada N segundos descarga
    precios, decide qué hacer y lo envía por un `Broker` (puerto de ypyquant.execution.broker; aquí el
    adaptador de Alpaca). La lógica de negocio vive fuera: cuándo toca rebalancear
    (execution/calendario.py), qué órdenes salen de unos pesos (execution/ejecutor.py, política del bot:
    acciones enteras, umbral 1 %, sin normalizar) y cómo se habla con Alpaca (execution/broker_alpaca.py).
    Aquí quedan el estado entre iteraciones y el cableado.

    Modo registro (`rebalancear_con_estrategia_registro`), reglas del bucle:
    - Historia: se descarga la que pide la estrategia (`Estrategia.sesiones_necesarias`), con el env
      como mínimo; si aun así falta, WARNING claro (la estrategia usa su caso de reserva y opera igual).
    - Frecuencia: solo se rebalancea cuando toca (diaria/semanal/mensual/trimestral, la de la estrategia
      o la indicada); la fecha del último rebalanceo se persiste (estado_bot.py) y sobrevive a reinicios.
      Los demás días solo se loguea.
    - Órdenes abiertas: no se envía orden nueva sobre un ticker con una orden aún abierta, y si no se
      pueden consultar las abiertas no se envía nada (evita duplicar).
    - Fills: al enviar una orden su cantidad ejecutada es 0, así que solo se anota su id como pendiente (con el
      precio de referencia con que se decidió); `sincronizar_fills` consulta después cada pendiente y registra el
      fill REAL (una sola vez), enriquecido con perfil, experimento, tipo y `precio_referencia`.
    - Memoria de la estrategia: `Estrategia.estado` se restaura al arrancar y se guarda tras cada decisión
      (estado_bot.py), así un reinicio no hace perder las posiciones de una `EstrategiaSenales`.
    - Instantánea diaria de la cuenta (equity, efectivo, posiciones con cantidad y precio) por perfil, una fila por
      día (`ypyquant.perfiles_alpaca.almacen`): Alpaca no devuelve las posiciones pasadas y sin ella no se puede
      medir el descuadre de cartera. La fecha es la del reloj del bot (UTC).
    - Modos: `max_iteraciones=1` es el modo `diario` (una pasada y sale, pensado para el programador de tareas);
      sin límite es el modo `continuo` (bucle cada N segundos). El reloj es inyectable (`ahora`) para los tests.
    - Mercado cerrado: por defecto se calculan los pesos pero NO se envía nada ni se marca el rebalanceo. Con
      `permitir_cerrado` se envían igualmente (orden de mercado DAY, que Alpaca deja en cola hasta la apertura;
      hipótesis a comprobar a mano, ver la checklist de docs/ui.md); los fills se registran al sincronizar.

Se relaciona con:
    - config.py: claves y parámetros.
    - precios.py: datos de entrada y cuánta historia pedir; estado_bot.py: último rebalanceo y órdenes pendientes.
    - ypyquant.execution: calendario, ejecutor, broker, broker_alpaca; nautilus.constructor_senales
      (modo señales), nautilus.ordenes (`decidir_orden`) y nautilus.registro_fills.
    - ypyquant.estrategias: modo registro (`estrategia_registro`): pesos de una Estrategia -> rebalanceo.
    - alpaca-py (extra `trading`): solo a través de BrokerAlpaca y de `crear_clientes`.
"""

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime

from pipelines.paper_trading.config import ConfiguracionEstrategia, ConfiguracionServicio
from pipelines.paper_trading.estado_bot import EstadoBot, guardar_estado_estrategia, restaurar_estado_estrategia
from pipelines.paper_trading.precios import (
    avisar_historia_insuficiente,
    descargar_precios,
    dias_naturales_para_sesiones,
    resolver_dias_historico,
)
from ypyquant.execution.broker import Broker
from ypyquant.execution.broker_alpaca import BrokerAlpaca
from ypyquant.execution.calendario import FRECUENCIAS_VALIDAS, toca_rebalancear
from ypyquant.execution.ejecutor import POLITICA_BOT, planificar_rebalanceo
from ypyquant.execution.nautilus.constructor_senales import construir_combinador
from ypyquant.execution.nautilus.ordenes import decidir_orden
from ypyquant.execution.nautilus.registro_fills import RegistroFills

logger = logging.getLogger(__name__)

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


def _abiertas_o_none(broker: Broker) -> set[str] | None:
    """Tickers con orden abierta, o None (con WARNING) si no se pudo consultar: entonces no se envía nada."""
    try:
        return broker.tickers_con_orden_abierta()
    except Exception as e:  # noqa: BLE001 - cualquier fallo del broker bloquea el envío, no el bucle
        logger.warning("No se pudieron consultar las órdenes abiertas (%s): no se envían órdenes para no duplicarlas", e)
        return None


def sincronizar_fills(broker: Broker, registro: RegistroFills, estado: EstadoBot,
                      reintentos: int = REINTENTOS_FILLS, espera_s: float = ESPERA_FILLS_S) -> int:
    """
    Registra el fill real de las órdenes enviadas que ya han terminado y las quita de las pendientes.

    Al enviar una orden de mercado su cantidad ejecutada es 0 (aún no se ha ejecutado), así que el fill no
    puede anotarse entonces: se guarda su id (`estado.ordenes_pendientes`) y aquí se consulta con el broker.
    Una orden en estado terminal (filled, canceled, expired, rejected...) con cantidad ejecutada > 0 se
    registra una sola vez (también el fill parcial de una cancelada); sin cantidad se descarta. Las aún
    abiertas (new, accepted, partially_filled) o que no se pudieron consultar siguen pendientes. Con
    `reintentos` > 0 espera `espera_s` entre consultas, para dar tiempo a que las órdenes de mercado recién
    enviadas se ejecuten. Devuelve cuántos fills se registraron.
    """
    registrados = 0
    for intento in range(reintentos + 1):
        for id_orden in list(estado.ordenes_pendientes):
            try:
                orden = broker.consultar_orden(id_orden)
            except Exception as e:  # noqa: BLE001 - se reintenta en la siguiente sincronización
                logger.warning("No se pudo consultar la orden %s: %s", id_orden, e)
                continue
            if not orden.terminal:
                continue
            if orden.cantidad_ejecutada > 0:
                if registro.registrar(orden.a_dict_fill(), {"precio_referencia": estado.referencias.get(str(id_orden))}) is not None:
                    registrados += 1
                    logger.info("FILL registrado: %s %s %s @ %s", orden.ticker, orden.lado,
                                orden.cantidad_ejecutada, orden.precio_medio)
            else:
                logger.info("Orden %s de %s terminó sin ejecutarse (%s)", id_orden, orden.ticker, orden.estado)
            estado.quitar_pendiente(id_orden)
        if not estado.ordenes_pendientes or intento == reintentos:
            break
        time.sleep(espera_s)
    return registrados


def _sincronizar_sin_fallar(broker: Broker, registro: RegistroFills, estado: EstadoBot, **opciones) -> None:
    """Sincroniza fills sin que un error de red tumbe la iteración (se reintenta en la siguiente)."""
    try:
        sincronizar_fills(broker, registro, estado, **opciones)
    except Exception as e:  # noqa: BLE001
        logger.warning("No se pudieron sincronizar los fills: %s", e)


def registrar_instantanea(broker: Broker, guardar: Callable[..., None], fecha: date) -> bool:
    """
    Guarda la foto de la cuenta del día: equity, efectivo y posiciones `{ticker: {cantidad, precio}}`.

    `guardar(fecha_iso, equity, efectivo, posiciones)` es normalmente `AlmacenPerfiles.guardar_instantanea` del perfil (una
    fila por día: si ya había una de hoy, la reemplaza). El precio de cada posición es la cotización del broker (None se
    guarda como 0.0: se ve que faltó). Nunca lanza: un fallo de red se avisa y se reintenta en la siguiente iteración.
    """
    try:
        cuenta = broker.estado_cuenta()
        posiciones = {}
        for ticker, cantidad in cuenta.posiciones.items():
            try:
                precio = broker.precio_actual(ticker) or 0.0
            except Exception as e:  # noqa: BLE001 - una cotización fallida no impide guardar el resto
                logger.warning("Sin cotización de %s para la instantánea: %s", ticker, e)
                precio = 0.0
            posiciones[ticker] = {"cantidad": float(cantidad), "precio": float(precio)}
        guardar(fecha.isoformat(), cuenta.equity, cuenta.efectivo, posiciones)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("No se pudo guardar la instantánea de la cuenta: %s", e)
        return False


def evaluar_y_enviar(broker: Broker, ticker, senal, estrategia, equity, cantidad_actual, abierto):
    """Modo señales: precio actual, decisión por señal (`decidir_orden`) y envío. Devuelve el id de orden o None."""
    if not abierto:
        return None
    try:
        precio = broker.precio_actual(ticker) or 0.0
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
        id_orden = broker.enviar_orden(ticker, decision.lado, decision.cantidad)
        logger.info("ORDEN %s: %d %s | señal=%.3f | precio≈%.2f",
                    decision.lado.upper(), decision.cantidad, ticker, senal, precio)
        return id_orden
    except Exception as e:  # noqa: BLE001 - una orden fallida no aborta el resto
        logger.warning("No se pudo evaluar/enviar la orden de %s: %s", ticker, e)
        return None


def precios_a_ohlcv(precios) -> dict:
    """DataFrame fecha x ticker de cierres -> `{ticker: OHLCV mínimo}` que esperan las estrategias del registro."""
    import pandas as pd

    return {t: pd.DataFrame({"Close": precios[t], "adjusted_close": precios[t]}).dropna() for t in precios.columns}


def rebalancear_con_estrategia_registro(broker: Broker, estrategia_registrada, precios, abierto, estado: EstadoBot,
                                        frecuencia: str, hoy: date | None = None, permitir_cerrado: bool = False):
    """
    Una iteración en modo registro: si toca, pide pesos a la estrategia y envía las órdenes de rebalanceo.

    `estrategia_registrada` es una instancia de `ypyquant.estrategias.Estrategia` (mantiene su `estado`
    entre iteraciones). Reglas:
    - Solo rebalancea cuando `toca_rebalancear(frecuencia, estado.ultimo_rebalanceo, hoy)`; si no, loguea
      y devuelve None (no calcula pesos ni envía nada).
    - Con el mercado cerrado calcula y muestra los pesos pero no envía ni marca el rebalanceo (salvo `permitir_cerrado`:
      envía igualmente órdenes DAY, que quedan en cola en el broker hasta la apertura).
    - No envía orden sobre un ticker con una orden abierta; si no puede consultar las abiertas, no envía nada.
    - Cada orden enviada queda como pendiente de registrar (su fill llega después, ver `sincronizar_fills`).
    - Marca el rebalanceo como hecho solo si ningún envío falló (si falla uno, se reintenta en la siguiente
      iteración; las ya enviadas quedan protegidas por el bloqueo de órdenes abiertas).

    Devuelve los pesos calculados, o None si hoy no tocaba.
    """
    hoy = hoy or datetime.now(UTC).date()
    if not toca_rebalancear(frecuencia, estado.ultimo_rebalanceo, hoy):
        logger.info("Hoy no toca rebalancear (frecuencia %s, último rebalanceo %s)", frecuencia, estado.ultimo_rebalanceo)
        return None

    datos = precios_a_ohlcv(precios)
    pesos = estrategia_registrada.como_funcion()(precios.index[-1].to_pydatetime(), datos)
    logger.info("Pesos objetivo de %s: %s", estrategia_registrada.nombre, {t: round(w, 4) for t, w in pesos.items()})
    if not abierto and not permitir_cerrado:
        logger.info("Mercado cerrado: no se enviarán órdenes")
        return pesos
    if not abierto:
        logger.info("Mercado cerrado: se envían las órdenes igualmente (quedarán en cola hasta la apertura)")
    bloqueados = _abiertas_o_none(broker)
    if bloqueados is None:
        return pesos
    cuenta = broker.estado_cuenta()
    ultimos = {t: float(precios[t].dropna().iloc[-1]) for t in precios.columns if precios[t].notna().any()}
    afectados = sorted(bloqueados & (set(pesos) | set(cuenta.posiciones)))
    if afectados:
        logger.info("No se envían órdenes nuevas para %s: ya hay una orden abierta sobre cada uno", ", ".join(afectados))

    hubo_error = False
    plan = planificar_rebalanceo(pesos, ultimos, cuenta, POLITICA_BOT, tickers_bloqueados=bloqueados)
    for orden in plan.ordenes:
        try:
            id_orden = broker.enviar_orden(orden.ticker, orden.lado, int(orden.cantidad))
            logger.info("ORDEN %s: %d %s (objetivo %d)", orden.lado.upper(), orden.cantidad, orden.ticker,
                        orden.cantidad_objetivo)
            estado.anadir_pendiente(id_orden, ultimos.get(orden.ticker))  # el fill real se registra en `sincronizar_fills`
        except Exception as e:  # noqa: BLE001 - una orden fallida no aborta el resto
            hubo_error = True
            logger.warning("No se pudo enviar la orden de %s: %s", orden.ticker, e)
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
    firma = f"{modo}|{parametros}|{','.join(sorted(estrategia.tickers))}"
    return f"{firma}|{estrategia.huella_receta}" if estrategia.huella_receta else firma


def _completar_equity_inicial(servicio: ConfiguracionServicio, equity: float) -> None:
    """Si el vínculo del perfil aún no tiene el equity inicial de la cuenta (se crea antes de abrir el broker), lo anota."""
    if servicio.almacen_perfiles is None or servicio.perfil is None:
        return
    vinculo = servicio.almacen_perfiles.leer_vinculo(servicio.perfil)
    if vinculo and vinculo.get("equity_inicial") is None:
        servicio.almacen_perfiles.guardar_vinculo(servicio.perfil, {**vinculo, "equity_inicial": equity})
        logger.info("Vínculo: equity inicial de la cuenta %s = %s", servicio.perfil, f"{equity:,.2f}")


def ejecutar_vivo(servicio: ConfiguracionServicio, estrategia: ConfiguracionEstrategia,
                  control: ControlParada, max_iteraciones: int | None = None,
                  ahora: Callable[[], datetime] | None = None, permitir_cerrado: bool = False):
    """
    Bucle principal de paper trading. `max_iteraciones=1` es el modo diario (una pasada y sale).

    `ahora` (reloj inyectable, UTC) y `permitir_cerrado` (enviar con el mercado cerrado) ver la cabecera del módulo.
    """
    ahora = ahora or (lambda: datetime.now(UTC))
    cliente, cliente_datos = crear_clientes(servicio)
    broker = BrokerAlpaca(cliente, cliente_datos)
    registro = RegistroFills(servicio.ruta_fills, servicio.etiquetas_fills)
    firma = _firma_estrategia(estrategia)
    estado = EstadoBot.cargar(servicio.ruta_estado, firma)
    guardar_instantanea = (None if servicio.almacen_perfiles is None or servicio.perfil is None
                           else lambda *args: servicio.almacen_perfiles.guardar_instantanea(servicio.perfil, *args))

    cuenta = broker.estado_cuenta()
    logger.info("Cuenta Alpaca%s | efectivo: %s | equity: %s", f" [{servicio.perfil}]" if servicio.perfil else "",
                f"{cuenta.efectivo:,.2f}", f"{cuenta.equity:,.2f}")
    _completar_equity_inicial(servicio, cuenta.equity)

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
        if servicio.ruta_estado_estrategia is not None:
            restaurar_estado_estrategia(servicio.ruta_estado_estrategia, firma, estrategia_registrada)
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
        hoy = ahora().date()
        logger.info("--- Iteración %d [%s] ---", iteracion, ahora().strftime("%H:%M:%S UTC"))
        _sincronizar_sin_fallar(broker, registro, estado, reintentos=0)  # fills de órdenes de iteraciones anteriores
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
                        try:
                            rebalancear_con_estrategia_registro(
                                broker, estrategia_registrada, precios, broker.mercado_abierto(), estado, frecuencia,
                                hoy=hoy, permitir_cerrado=permitir_cerrado)
                        finally:  # la estrategia avanzó su memoria aunque algún envío fallara: se guarda siempre
                            if servicio.ruta_estado_estrategia is not None:
                                guardar_estado_estrategia(servicio.ruta_estado_estrategia, firma, estrategia_registrada)
            else:
                precios = descargar_precios(estrategia.tickers, dias_historico)
                senales = combinador.generate(precios) if not precios.empty else None
                if senales is None or senales.empty:
                    logger.warning("Sin precios o sin señales; se reintenta en la siguiente iteración")
                else:
                    ultimas = senales.iloc[-1]
                    logger.info("Señales actuales: %s", ultimas.to_dict())
                    cuenta = broker.estado_cuenta()
                    abierto = broker.mercado_abierto()
                    if not abierto:
                        logger.info("Mercado cerrado: no se enviarán órdenes")
                    bloqueados = _abiertas_o_none(broker) if abierto else set()
                    for ticker in estrategia.tickers:
                        if ticker not in ultimas.index or bloqueados is None:
                            continue
                        if ticker in bloqueados:
                            logger.info("No se envía orden para %s: ya hay una orden abierta", ticker)
                            continue
                        id_orden = evaluar_y_enviar(broker, ticker, float(ultimas[ticker]), estrategia, cuenta.equity,
                                                    int(cuenta.posiciones.get(ticker, 0)), abierto)
                        if id_orden is not None:
                            estado.anadir_pendiente(id_orden)
        except Exception as e:
            logger.error("Error en la iteración %d: %s", iteracion, e, exc_info=True)
        # Las órdenes de mercado recién enviadas suelen ejecutarse en segundos: se registra su fill real ahora
        _sincronizar_sin_fallar(broker, registro, estado, reintentos=REINTENTOS_FILLS, espera_s=ESPERA_FILLS_S)
        if guardar_instantanea is not None:
            registrar_instantanea(broker, guardar_instantanea, hoy)

        if control.en_marcha and (max_iteraciones is None or iteracion < max_iteraciones):
            control.dormir(servicio.segundos_entre_iteraciones)

    logger.info("Servicio detenido limpiamente")
