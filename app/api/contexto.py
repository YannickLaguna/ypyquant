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

from app.api.gestor_bots import GestorBots
from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.data.matriz_precios import CargadorPrecios
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.barridos import AlmacenBarridos
from ypyquant.estrategias.configuraciones import AlmacenConfiguraciones
from ypyquant.estrategias.marcos import AlmacenMarcos
from ypyquant.estrategias.servicio_experimentos import ServicioExperimentos
from ypyquant.perfiles_alpaca.almacen import AlmacenPerfiles
from ypyquant.perfiles_alpaca.perfiles import nombre_valido
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
    # Alpaca multi-perfil: `alpacas` = un `ServicioAlpaca` por perfil (`alpaca` es el perfil DEFECTO y se registra solo;
    # el resto se crean al pedirlos con `alpaca_de`), `almacen_perfiles` = datos de cada perfil en disco y
    # `gestor_bots` = un bot por perfil (ver app/api/gestor_bots.py)
    alpacas: dict[str, ServicioAlpaca] = field(default_factory=dict)
    almacen_perfiles: AlmacenPerfiles | None = None
    gestor_bots: GestorBots | None = None

    def __post_init__(self) -> None:
        raiz = RaizDatos.de_experimentos(self.almacen.raiz)
        if self.almacen_barridos is None:
            self.almacen_barridos = AlmacenBarridos(raiz.barridos, self.almacen, raiz.reservado)
        if self.almacen_configuraciones is None:
            self.almacen_configuraciones = AlmacenConfiguraciones(raiz.configuraciones)
        if self.almacen_marcos is None:
            self.almacen_marcos = AlmacenMarcos(raiz.marcos)
        if self.almacen_perfiles is None:
            self.almacen_perfiles = AlmacenPerfiles(raiz.paper_trading)
        if self.gestor_bots is None:
            self.gestor_bots = GestorBots(self.almacen_perfiles)
        self.alpacas.setdefault(self.alpaca.perfil, self.alpaca)
        if self.servicio is None:
            self.servicio = ServicioExperimentos(self.almacen, self.cargador_precios, self.ejecutor,
                                                 self.almacen_barridos, self.almacen_marcos, self.ejecutor_barridos)


def obtener_contexto(request: Request) -> Contexto:
    """Dependencia de FastAPI: devuelve el contexto de la aplicación."""
    return request.app.state.ctx


def alpaca_de(ctx: Contexto, perfil: str) -> ServicioAlpaca:
    """Servicio de Alpaca del perfil (lo crea la primera vez). `ValueError` si el nombre del perfil no es válido."""
    if not nombre_valido(perfil):
        raise ValueError(f"Nombre de perfil inválido: {perfil!r} (mayúsculas, dígitos y _)")
    if perfil not in ctx.alpacas:
        ctx.alpacas[perfil] = ServicioAlpaca(perfil=perfil)
    return ctx.alpacas[perfil]
