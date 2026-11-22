"""
__main__.py — CLI del paper trading.

Uso (desde la raíz del repo, con el extra `trading` instalado):
    uv sync --extra trading
    uv run python -m pipelines.paper_trading --backtest
    uv run python -m pipelines.paper_trading --una-vez
    uv run python -m pipelines.paper_trading --config pipelines/paper_trading/estrategia_ejemplo.json
    # Estrategia del registro de ypyquant (incluidas o de estrategias_usuario/):
    uv run python -m pipelines.paper_trading --estrategia momentum --parametros '{"top_k": 3}' --tickers AAPL,MSFT,NVDA
    # Con una historia mínima mayor y rebalanceo semanal en vez del mensual que sugiere momentum:
    uv run python -m pipelines.paper_trading --estrategia momentum --dias-historico 600 --frecuencia weekly

Modo multi-perfil (una cuenta paper de Alpaca por perfil, cada una operando UN experimento; ver docs/ui.md):
    # Vincula la cuenta B al experimento (congela su receta) y opera en bucle (continuo) o una pasada (diario)
    uv run python -m pipelines.paper_trading --perfil B --experimento momentum-1a2b3c4d5e --modo diario
    # Sin --experimento reanuda el vínculo guardado en data/paper_trading/B/vinculo.json (p. ej. desde el programador)
    uv run python -m pipelines.paper_trading --perfil B --modo diario
    # --configuracion ID en vez de --experimento toma la receta de una configuración guardada

Las claves del perfil son ALPACA_<PERFIL>_API_KEY / ALPACA_<PERFIL>_SECRET_KEY (DEFECTO = ALPACA_API_KEY / ALPACA_SECRET_KEY).
Los logs pasan por un filtro de redacción: ninguna clave sale por la salida estándar.

Se relaciona con: config.py, alpaca_vivo.py, backtest.py (este fichero solo parsea argumentos y orquesta),
ypyquant.estrategias.base (frecuencias válidas) y ypyquant.perfiles_alpaca (vínculo, almacén por perfil, redacción).
"""

import argparse
import json
import logging
import signal
from pathlib import Path

from pipelines.paper_trading.config import (
    ConfiguracionEstrategia,
    ConfiguracionServicio,
    cargar_env,
)
from ypyquant.execution.calendario import FRECUENCIAS_VALIDAS
from ypyquant.perfiles_alpaca.perfiles import capacidad_declarada, nombre_valido
from ypyquant.perfiles_alpaca.redaccion import instalar_en_raiz
from ypyquant.raiz_datos import RaizDatos

FORMATO_LOG = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipelines.paper_trading",
                                     description="Paper trading personal con ypyquant + Alpaca (cuenta paper)")
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON con la estrategia (ver estrategia_ejemplo.json); por defecto, técnica sobre 5 tickers")
    parser.add_argument("--estrategia", default=None,
                        help="Nombre de una estrategia del registro de ypyquant (modo pesos; ver docs/ui.md)")
    parser.add_argument("--parametros", default=None, help="JSON con los parámetros de --estrategia")
    parser.add_argument("--tickers", default=None, help="Tickers separados por comas (sobrescribe la configuración)")
    parser.add_argument("--backtest", action="store_true",
                        help="Backtest de las señales con el motor de ypyquant (no necesita claves de Alpaca)")
    parser.add_argument("--una-vez", action="store_true", help="Una sola iteración en vivo y salir (= --modo diario)")
    parser.add_argument("--perfil", default=None,
                        help="Perfil de la cuenta de Alpaca (ALPACA_<PERFIL>_API_KEY / _SECRET_KEY; DEFECTO = las variables sin nombre). "
                             "Con --perfil el bot opera el experimento vinculado a esa cuenta")
    parser.add_argument("--experimento", default=None, help="Id del experimento al que vincular el perfil (congela su receta)")
    parser.add_argument("--configuracion", default=None, help="Id de una configuración guardada, en vez de --experimento")
    parser.add_argument("--modo", choices=("diario", "continuo"), default=None,
                        help="diario = una pasada y salir (para el programador de tareas); continuo = bucle cada --intervalo. "
                             "Con --perfil el defecto es continuo")
    parser.add_argument("--enviar-fuera-de-horario", action="store_true",
                        help="Envía las órdenes aunque el mercado esté cerrado (DAY: quedan en cola hasta la apertura; sin verificar, "
                             "ver la checklist de docs/ui.md)")
    parser.add_argument("--datos", type=Path, default=None, help=argparse.SUPPRESS)  # carpeta data/ alternativa (tests)
    parser.add_argument("--intervalo", type=int, default=None, help="Segundos entre iteraciones (sobrescribe el entorno)")
    parser.add_argument("--dias-historico", type=int, default=None,
                        help="Mínimo de días naturales de historia a descargar (sobrescribe NAUTILUS_LOOKBACK_DAYS). En modo "
                             "registro se descargan más si la estrategia los necesita (momentum 12-1, ventanas del TFM...)")
    parser.add_argument("--frecuencia", choices=FRECUENCIAS_VALIDAS, default=None,
                        help="Frecuencia de rebalanceo en modo registro; por defecto la que sugiere la estrategia")
    return parser


def preparar_perfil(args: argparse.Namespace) -> tuple[ConfiguracionServicio, ConfiguracionEstrategia]:
    """
    Servicio y estrategia del bot de un perfil, a partir de su vínculo con un experimento.

    Con `--experimento` / `--configuracion` crea (o reconfirma) el vínculo: congela la receta en `vinculo.json`. Sin ellos
    reanuda el vínculo guardado. `SystemExit` con un mensaje claro si falta algo (perfil inválido, sin vínculo, experimento
    inexistente, universo no operable). El equity inicial lo completa el bucle al abrir la cuenta.
    """
    from pydantic import ValidationError

    from ypyquant.estrategias.almacen import AlmacenExperimentos
    from ypyquant.estrategias.configuraciones import AlmacenConfiguraciones
    from ypyquant.perfiles_alpaca.almacen import AlmacenPerfiles
    from ypyquant.perfiles_alpaca.vinculo import UniversoNoSoportado, receta_de_origen, vincular

    perfil = args.perfil
    if not nombre_valido(perfil):
        raise SystemExit(f"Nombre de perfil inválido: {perfil!r} (mayúsculas, dígitos y _)")
    if args.estrategia or args.config or args.tickers:
        raise SystemExit("Con --perfil la estrategia sale del experimento vinculado: no uses --estrategia, --config ni --tickers")
    if args.experimento and args.configuracion:
        raise SystemExit("Indica solo uno de --experimento o --configuracion")
    raiz = RaizDatos(base=args.datos) if args.datos else RaizDatos()
    almacen = AlmacenPerfiles(raiz.paper_trading)
    if args.experimento or args.configuracion:
        tipo, id_origen = ("experimento", args.experimento) if args.experimento else ("configuracion", args.configuracion)
        try:
            receta = receta_de_origen(tipo, id_origen, AlmacenExperimentos(raiz.experimentos),
                                      AlmacenConfiguraciones(raiz.configuraciones))
            vinculo, cambio = vincular(almacen, perfil, receta, tipo, id_origen, None, capacidad=capacidad_declarada(perfil))
        except (KeyError, ValueError, UniversoNoSoportado, ValidationError) as error:
            raise SystemExit(f"No se puede vincular el perfil {perfil} a {tipo} '{id_origen}': {error}") from error
        logging.getLogger(__name__).info("Perfil %s %s a %s '%s' (estrategia %s, %d tickers)", perfil,
                                         "vinculado" if cambio else "ya vinculado", tipo, id_origen, vinculo["estrategia"],
                                         len(vinculo["tickers"]))
    else:
        vinculo = almacen.leer_vinculo(perfil)
        if vinculo is None:
            raise SystemExit(f"El perfil {perfil} no está vinculado a ningún experimento: usa --experimento ID")
    estrategia = ConfiguracionEstrategia.desde_vinculo(vinculo)
    servicio = ConfiguracionServicio.desde_entorno(perfil=perfil, raiz_datos=raiz)
    servicio.etiquetas_fills[f"{vinculo['origen']['tipo']}_id"] = vinculo["origen"]["id"]
    return servicio, estrategia


def main(argv: list[str] | None = None) -> None:
    args = crear_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=FORMATO_LOG, datefmt="%Y-%m-%d %H:%M:%S")
    instalar_en_raiz()  # ninguna clave de Alpaca llega a la salida (que el gestor de bots guarda en bot.log)
    for ruidosa in ("yfinance", "urllib3", "httpx"):
        logging.getLogger(ruidosa).setLevel(logging.WARNING)
    cargar_env()

    if args.perfil:
        if args.backtest:
            raise SystemExit("--backtest no se combina con --perfil")
        servicio, estrategia = preparar_perfil(args)
        if args.intervalo:
            servicio.segundos_entre_iteraciones = args.intervalo
        if args.dias_historico:
            servicio.dias_historico = args.dias_historico
        una_pasada = args.una_vez or args.modo == "diario"
        if not una_pasada:
            logging.getLogger(__name__).warning(
                "Modo continuo: el bucle consulta cada %ds, pero solo rebalancea cuando toca la frecuencia (%s)",
                servicio.segundos_entre_iteraciones, estrategia.frecuencia_rebalanceo_registro)
        from pipelines.paper_trading.alpaca_vivo import ControlParada, ejecutar_vivo
        control = ControlParada()
        signal.signal(signal.SIGINT, control.detener)
        signal.signal(signal.SIGTERM, control.detener)
        ejecutar_vivo(servicio, estrategia, control, max_iteraciones=1 if una_pasada else None,
                      permitir_cerrado=args.enviar_fuera_de_horario)
        return
    if args.experimento or args.configuracion:
        raise SystemExit("--experimento y --configuracion necesitan --perfil (la cuenta a vincular)")

    estrategia = (ConfiguracionEstrategia.desde_json(args.config) if args.config
                  else ConfiguracionEstrategia())

    if args.estrategia:
        estrategia.estrategia_registro = args.estrategia
        estrategia.nombre = args.estrategia
        estrategia.parametros_registro = json.loads(args.parametros) if args.parametros else {}
    if args.frecuencia:
        estrategia.frecuencia_rebalanceo_registro = args.frecuencia
    if args.tickers:
        estrategia.tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.backtest:
        from pipelines.paper_trading.backtest import ejecutar_backtest
        ejecutar_backtest(estrategia)
        return

    from pipelines.paper_trading.alpaca_vivo import ControlParada, ejecutar_vivo
    servicio = ConfiguracionServicio.desde_entorno()
    if args.intervalo:
        servicio.segundos_entre_iteraciones = args.intervalo
    if args.dias_historico:
        servicio.dias_historico = args.dias_historico
    control = ControlParada()
    signal.signal(signal.SIGINT, control.detener)
    signal.signal(signal.SIGTERM, control.detener)
    ejecutar_vivo(servicio, estrategia, control, max_iteraciones=1 if args.una_vez or args.modo == "diario" else None,
                  permitir_cerrado=args.enviar_fuera_de_horario)


if __name__ == "__main__":
    main()
