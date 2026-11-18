"""
Contexto compartido de la API: almacén, cargador de precios, ejecutor de experimentos y servicios de paper trading.

Responsabilidad:
- `Contexto`: agrupa las dependencias que las rutas necesitan (se guarda en `app.state.ctx`).
  Así los tests inyectan precios sintéticos, un almacén temporal y un cliente Alpaca falso.
- `lanzar_experimento`: registra el experimento como `pendiente` y lo ejecuta en un hilo
  (`ThreadPoolExecutor`), dejando el estado y la traza de error en el almacén. La API no tiene
  lógica de negocio: delega todo en `ypyquant.estrategias`.

Se relaciona con:
- app/api/main.py (crea el contexto), app/api/rutas_*.py (lo consumen vía `obtener_contexto`).
- ypyquant.estrategias (ejecutar_experimento, AlmacenExperimentos, cargadores de precios).
- app/api/paper.py (servicios de paper trading que cuelgan del contexto).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from fastapi import Request

from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.data.matriz_precios import CargadorPrecios
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, ejecutar_experimento

logger = logging.getLogger(__name__)


@dataclass
class Contexto:
    """Dependencias de la API. `ejecutor` corre los experimentos en segundo plano (2 a la vez por defecto)."""

    almacen: AlmacenExperimentos
    cargador_precios: CargadorPrecios
    alpaca: ServicioAlpaca
    bot: BotControl
    ejecutor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=2, thread_name_prefix="exp"))


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
