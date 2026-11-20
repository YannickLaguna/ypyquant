"""
Contexto compartido de la API: almacén, cargador de precios, ejecutor de experimentos y servicios de paper trading.

Responsabilidad:
- `Contexto`: agrupa las dependencias que las rutas necesitan (se guarda en `app.state.ctx`).
  Así los tests inyectan precios sintéticos, un almacén temporal y un cliente Alpaca falso.
- `lanzar_experimento`: registra el experimento como `pendiente` y lo ejecuta en un hilo
  (`ThreadPoolExecutor`), dejando el estado y la traza de error en el almacén. La API no tiene
  lógica de negocio: delega todo en `ypyquant.estrategias`.
- Barridos: `lanzar_barrido` (crea el barrido y encola todas sus combinaciones como experimentos) y
  `lanzar_permutacion` (prueba de permutación en un hilo aparte, con estado/progreso en
  `barrido.json` -> `tareas` y cancelación cooperativa con un `threading.Event` por tarea).

Se relaciona con:
- app/api/main.py (crea el contexto), app/api/rutas_*.py (lo consumen vía `obtener_contexto`).
- ypyquant.estrategias (ejecutar_experimento, AlmacenExperimentos, AlmacenBarridos,
  prueba_permutacion_barrido, cargadores de precios).
- app/api/paper.py (servicios de paper trading que cuelgan del contexto).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.data.matriz_precios import CargadorPrecios
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.barridos import AlmacenBarridos, ConfiguracionBarrido
from ypyquant.estrategias.configuraciones import AlmacenConfiguraciones
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, ejecutar_experimento
from ypyquant.estrategias.marcos import AlmacenMarcos
from ypyquant.estrategias.validacion_barridos import PruebaCancelada, prueba_permutacion_barrido

logger = logging.getLogger(__name__)


@dataclass
class Contexto:
    """
    Dependencias de la API. `ejecutor` corre los experimentos en segundo plano (2 a la vez por
    defecto) y `ejecutor_barridos` las pruebas de permutación (1 a la vez: ya paralelizan por dentro).
    `almacen_barridos` (por defecto `<carpeta del almacén>/../barridos`) guarda los barridos y
    `cancelaciones` los eventos de cancelación de las tareas en curso, por `(id_barrido, tarea)`.
    `almacen_configuraciones` (`<carpeta del almacén>/../configuraciones`) y `almacen_marcos` (`../marcos`)
    guardan las configuraciones versionadas y los marcos de comparación con sus comparativas.
    """

    almacen: AlmacenExperimentos
    cargador_precios: CargadorPrecios
    alpaca: ServicioAlpaca
    bot: BotControl
    ejecutor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=2, thread_name_prefix="exp"))
    almacen_barridos: AlmacenBarridos | None = None
    ejecutor_barridos: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="barrido"))
    cancelaciones: dict[tuple[str, str], threading.Event] = field(default_factory=dict)
    almacen_configuraciones: AlmacenConfiguraciones | None = None
    almacen_marcos: AlmacenMarcos | None = None

    def __post_init__(self) -> None:
        if self.almacen_barridos is None:
            self.almacen_barridos = AlmacenBarridos(self.almacen.raiz.parent / "barridos", self.almacen)
        if self.almacen_configuraciones is None:
            self.almacen_configuraciones = AlmacenConfiguraciones(self.almacen.raiz.parent / "configuraciones")
        if self.almacen_marcos is None:
            self.almacen_marcos = AlmacenMarcos(self.almacen.raiz.parent / "marcos")


def obtener_contexto(request: Request) -> Contexto:
    """Dependencia de FastAPI: devuelve el contexto de la aplicación."""
    return request.app.state.ctx


def _ejecutar_en_segundo_plano(ctx: Contexto, id_experimento: str, config: ConfiguracionExperimento) -> None:
    """Cuerpo del hilo: en_curso -> hecho | error (con traza). Nunca lanza."""
    try:
        ctx.almacen.marcar_estado(id_experimento, "en_curso")
        resultado = ejecutar_experimento(config, ctx.cargador_precios)
        ctx.almacen.guardar(id_experimento, resultado)
    except BaseException as error:
        logger.exception("Experimento %s falló", id_experimento)
        ctx.almacen.marcar_estado(id_experimento, "error", error)


def lanzar_experimento(ctx: Contexto, config: ConfiguracionExperimento, forzar: bool = False) -> tuple[str, bool]:
    """
    Registra y encola un experimento. Devuelve `(id, reutilizado)`.

    Si ya existe uno con la misma huella terminado correctamente (o en curso) y no se fuerza,
    se reutiliza en vez de recalcularlo (misma configuración = mismo resultado).
    """
    config = config.normalizada()
    id_experimento = config.identificador()
    if not forzar and ctx.almacen.existe(id_experimento):
        estado = ctx.almacen.leer(id_experimento)["estado"]
        if estado in ("hecho", "pendiente", "en_curso"):
            return id_experimento, True
    ctx.almacen.registrar(config)
    ctx.ejecutor.submit(_ejecutar_en_segundo_plano, ctx, id_experimento, config)
    return id_experimento, False


def lanzar_barrido(ctx: Contexto, config: ConfiguracionBarrido, forzar: bool = False) -> tuple[str, list[str], bool]:
    """
    Crea el barrido y encola todas sus combinaciones como experimentos. Devuelve `(id, ids, reutilizado)`.

    Si el barrido ya existía (misma configuración) se conserva con sus resultados y los experimentos
    se reutilizan; con `forzar` todo se recalcula.
    """
    id_barrido, creado = ctx.almacen_barridos.crear(config, forzar)
    ids = [lanzar_experimento(ctx, combinacion, forzar)[0] for combinacion, _, _ in config.combinaciones()]
    return id_barrido, ids, not creado


def nombre_tarea_permutacion(modo: str) -> str:
    """Nombre de la tarea en `barrido.json` -> `tareas` (`permutacion_in_sample` | `permutacion_walk_forward`)."""
    return f"permutacion_{modo}"


def _ejecutar_permutacion(ctx: Contexto, id_barrido: str, nombre: str, parametros: dict[str, Any],
                          evento: threading.Event) -> None:
    """Cuerpo del hilo de la permutación: en_curso -> hecho | cancelado | error, con progreso. Nunca lanza."""
    almacen = ctx.almacen_barridos
    try:
        almacen.actualizar_tarea(id_barrido, nombre, estado="en_curso")
        resultado = prueba_permutacion_barrido(
            almacen.cargar(id_barrido), ctx.cargador_precios, cancelar=evento,
            callback_progreso=lambda hechas, total: almacen.actualizar_tarea(
                id_barrido, nombre, progreso={"completadas": hechas, "total": total}),
            **parametros)
        almacen.actualizar_tarea(id_barrido, nombre, estado="hecho", p_valor=resultado["p_valor"])
    except PruebaCancelada:
        almacen.actualizar_tarea(id_barrido, nombre, estado="cancelado", error=None)
    except BaseException as error:
        logger.exception("Permutación de %s falló", id_barrido)
        almacen.actualizar_tarea(id_barrido, nombre, estado="error", error=f"{type(error).__name__}: {error}")
    finally:
        ctx.cancelaciones.pop((id_barrido, nombre), None)


def lanzar_permutacion(ctx: Contexto, id_barrido: str, parametros: dict[str, Any]) -> str:
    """
    Encola la prueba de permutación del barrido y devuelve el nombre de la tarea.

    `parametros` son los argumentos de `prueba_permutacion_barrido` (n_permutaciones, modo...).
    Lanza `RuntimeError` si ya hay una igual en curso.
    """
    nombre = nombre_tarea_permutacion(parametros["modo"])
    actual = ctx.almacen_barridos.tarea(id_barrido, nombre)
    if actual and actual.get("estado") in ("pendiente", "en_curso"):
        raise RuntimeError(f"Ya hay una {nombre} en curso para este barrido")
    ctx.almacen_barridos.actualizar_tarea(
        id_barrido, nombre, estado="pendiente", error=None, p_valor=None, parametros=parametros,
        progreso={"completadas": 0, "total": parametros["n_permutaciones"]})
    evento = threading.Event()
    ctx.cancelaciones[(id_barrido, nombre)] = evento
    ctx.ejecutor_barridos.submit(_ejecutar_permutacion, ctx, id_barrido, nombre, parametros, evento)
    return nombre
