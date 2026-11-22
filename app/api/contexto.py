"""
Contexto compartido de la API: almacén, cargador de precios, ejecutor de experimentos y servicios de paper trading.

Responsabilidad:
- `Contexto`: agrupa las dependencias que las rutas necesitan (se guarda en `app.state.ctx`).
  Así los tests inyectan precios sintéticos, un almacén temporal y un cliente Alpaca falso.
- `servicio` (`ServicioExperimentos`, de dominio): lanza experimentos, barridos, comparativas y pruebas de permutación
  (con su cancelación). Las rutas lo llaman y solo traducen sus excepciones a HTTP; la API no tiene lógica de negocio:
  delega todo en `ypyquant.estrategias`.
- Las rutas de los almacenes salen de una sola `RaizDatos` (`ypyquant.raiz_datos`): aquí no se recomponen.

Se relaciona con:
- app/api/main.py (crea el contexto), app/api/rutas_*.py (lo consumen vía `obtener_contexto`).
- ypyquant.raiz_datos (rutas de los almacenes) y ypyquant.estrategias (ServicioExperimentos, AlmacenExperimentos,
  AlmacenBarridos, cargadores de precios).
- app/api/paper.py (servicios de paper trading que cuelgan del contexto).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from fastapi import Request

from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.data.matriz_precios import CargadorPrecios
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.barridos import AlmacenBarridos
from ypyquant.estrategias.configuraciones import AlmacenConfiguraciones
from ypyquant.estrategias.marcos import AlmacenMarcos
from ypyquant.estrategias.servicio_experimentos import ServicioExperimentos
from ypyquant.raiz_datos import RaizDatos


@dataclass
class Contexto:
    """
    Dependencias de la API. `ejecutor` corre los experimentos en segundo plano (2 a la vez por
    defecto) y `ejecutor_barridos` las pruebas de permutación (1 a la vez: ya paralelizan por dentro).
    `almacen_barridos` guarda los barridos, `almacen_configuraciones` las configuraciones versionadas y `almacen_marcos`
    los marcos de comparación con sus comparativas; los que no se inyectan cuelgan de la misma carpeta base que el
    almacén de experimentos (`RaizDatos.de_experimentos`). `servicio` comparte `ejecutor` y `ejecutor_barridos`
    (reemplazar `ctx.ejecutor.submit` en un test también afecta al servicio).
    """

    almacen: AlmacenExperimentos
    cargador_precios: CargadorPrecios
    alpaca: ServicioAlpaca
    bot: BotControl
    ejecutor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=2, thread_name_prefix="exp"))
    almacen_barridos: AlmacenBarridos | None = None
    ejecutor_barridos: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="barrido"))
    almacen_configuraciones: AlmacenConfiguraciones | None = None
    almacen_marcos: AlmacenMarcos | None = None
    servicio: ServicioExperimentos | None = None  # se construye con los almacenes y el `ejecutor` de este contexto

    def __post_init__(self) -> None:
        raiz = RaizDatos.de_experimentos(self.almacen.raiz)
        if self.almacen_barridos is None:
            self.almacen_barridos = AlmacenBarridos(raiz.barridos, self.almacen, raiz.reservado)
        if self.almacen_configuraciones is None:
            self.almacen_configuraciones = AlmacenConfiguraciones(raiz.configuraciones)
        if self.almacen_marcos is None:
            self.almacen_marcos = AlmacenMarcos(raiz.marcos)
        if self.servicio is None:
            self.servicio = ServicioExperimentos(self.almacen, self.cargador_precios, self.ejecutor,
                                                 self.almacen_barridos, self.almacen_marcos, self.ejecutor_barridos)


def obtener_contexto(request: Request) -> Contexto:
    """Dependencia de FastAPI: devuelve el contexto de la aplicación."""
    return request.app.state.ctx
