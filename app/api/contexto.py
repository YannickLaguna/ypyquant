"""
Contexto compartido de la API: almacén, cargador de precios, ejecutor de experimentos y servicios de paper trading.

Responsabilidad:
- `Contexto`: agrupa las dependencias que las rutas necesitan (se guarda en `app.state.ctx`).
  Así los tests inyectan precios sintéticos, un almacén temporal y un cliente Alpaca falso.
- `servicio` (`ServicioExperimentos`, de dominio): lanza experimentos, barridos y comparativas. Las rutas lo llaman y
  solo traducen sus excepciones a HTTP; la API no tiene lógica de negocio: delega todo en `ypyquant.estrategias`.
- `lanzar_permutacion` (prueba de permutación en un hilo aparte, con estado/progreso en
  `barrido.json` -> `tareas` y cancelación cooperativa con un `threading.Event` por tarea).

Se relaciona con:
- app/api/main.py (crea el contexto), app/api/rutas_*.py (lo consumen vía `obtener_contexto`).
- ypyquant.estrategias (ServicioExperimentos, AlmacenExperimentos, AlmacenBarridos,
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
from ypyquant.estrategias.barridos import AlmacenBarridos
from ypyquant.estrategias.configuraciones import AlmacenConfiguraciones
from ypyquant.estrategias.marcos import AlmacenMarcos
from ypyquant.estrategias.servicio_experimentos import ServicioExperimentos
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
    guardan las configuraciones versionadas y los marcos de comparación con sus comparativas. `servicio` comparte
    `ejecutor` (reemplazar `ctx.ejecutor.submit` en un test también afecta al servicio).
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
    servicio: ServicioExperimentos | None = None  # se construye con los almacenes y el `ejecutor` de este contexto

    def __post_init__(self) -> None:
        if self.almacen_barridos is None:
            self.almacen_barridos = AlmacenBarridos(self.almacen.raiz.parent / "barridos", self.almacen)
        if self.almacen_configuraciones is None:
            self.almacen_configuraciones = AlmacenConfiguraciones(self.almacen.raiz.parent / "configuraciones")
        if self.almacen_marcos is None:
            self.almacen_marcos = AlmacenMarcos(self.almacen.raiz.parent / "marcos")
        if self.servicio is None:
            self.servicio = ServicioExperimentos(self.almacen, self.cargador_precios, self.ejecutor,
                                                 self.almacen_barridos, self.almacen_marcos)


def obtener_contexto(request: Request) -> Contexto:
    """Dependencia de FastAPI: devuelve el contexto de la aplicación."""
    return request.app.state.ctx


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
