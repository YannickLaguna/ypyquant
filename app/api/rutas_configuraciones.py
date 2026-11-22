"""
Rutas de configuraciones guardadas (recetas congeladas y versionadas de una estrategia).

Responsabilidad (prefijo `/api/configuraciones`):
- `GET /` (filtros `estrategia`, `texto`, `archivadas`): última versión de cada configuración.
- `POST /`: crea la versión 1 desde UNA de estas fuentes: `desde_experimento` (id de un experimento terminado),
  `desde_barrido` (+ `criterio` `mejor` | `meseta` | `elegida`, y `id_experimento` si es `elegida`) o `manual`
  (receta sin fechas). Guarda las métricas del experimento de origen EN ESE MOMENTO. 409 si el id ya existe.
- `GET /{id}?version=`: detalle con las versiones, el marco enlazado (si hay) y los backtests de la misma receta.
- `POST /{id}/versiones`: `{cambios}` crea la versión n+1 (la anterior no se toca).
- `POST /{id}/backtest`: `{fecha_inicio?, fecha_fin?, version?, capital_inicial?, forzar?}` lanza (o reutiliza) un
  experimento normal con `a_experimento`. Sin fechas: las del marco enlazado o los últimos 5 años.
- `POST /{id}/comparativa`: `{estrategias: [{estrategia, parametros}], fecha_inicio?, fecha_fin?, version?, forzar?}` lanza la
  MISMA configuración de entorno contra N estrategias a la vez. Convierte la configuración en un marco
  (`marco_desde_configuracion`: universo, frecuencia, costes, benchmark, capital, semilla, historia previa y los
  parámetros comunes que traiga), lo crea o reutiliza (id derivado de configuración + versión + fechas: idempotente) y
  usa el flujo de `POST /api/marcos/{id}/comparativa` (valida todas antes de lanzar, 202). Sin fechas: las del marco
  enlazado, si no las del experimento de origen y, si no, 5 años. El detalle de la configuración devuelve además
  `fechas_sugeridas` y las `comparativas` lanzadas así. La configuración NO cambia (ni versión nueva): el enlace
  vive en el marco (`id_configuracion`, `version_configuracion`).
- `DELETE /{id}`: archiva; con `?definitivo=true` borra todas las versiones.

Códigos: 404 configuración, experimento, barrido o estrategia inexistente; 409 id duplicado o experimento sin
terminar; 422 parámetros inválidos o fuentes incoherentes. Las rutas de paper interno y Alpaca (fases 3-5) se añadirán
aquí cuando existan.

Se relaciona con: app/api/contexto.py (almacenes y `ctx.servicio`), ypyquant.estrategias.configuraciones y
.barridos (`superficie_registros`), app/web (páginas Configuraciones).
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.api.contexto import Contexto, obtener_contexto
from app.api.rutas_marcos import PeticionComparativa, con_estado
from ypyquant.estrategias.barridos import superficie_registros
from ypyquant.estrategias.configuraciones import (
    ConfiguracionExistente,
    ConfiguracionGuardada,
    OrigenConfiguracion,
)
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, Frecuencia
from ypyquant.estrategias.marcos import fechas_por_defecto

router = APIRouter(prefix="/api/configuraciones", tags=["configuraciones"])

AVISO_BARRIDO = ("Elegida entre varias combinaciones de un barrido: puede estar sobreajustada (winner's curse). "
                 "Revisa DSR/PBO y la meseta.")


class RecetaManual(BaseModel):
    """Receta escrita a mano (sin fechas): lo mínimo es la estrategia y el universo."""

    estrategia: str
    parametros: dict[str, Any] = Field(default_factory=dict)
    universo: list[str] | str
    frecuencia_rebalanceo: Frecuencia | None = None
    comision: float = Field(0.001, ge=0, le=0.1)
    slippage: float = Field(0.001, ge=0, le=0.1)
    benchmark: str = "SPY"
    capital_inicial: float = Field(100_000.0, gt=0)
    semilla: int = 42
    dias_historia_previa: int = Field(1100, ge=0, le=7300)


class PeticionConfiguracion(BaseModel):
    """Nombre y notas más exactamente una fuente: `desde_experimento`, `desde_barrido` o `manual`."""

    nombre: str = Field(min_length=1, max_length=120)
    notas: str = ""
    id: str | None = None
    etiquetas: list[str] = Field(default_factory=list)
    id_marco: str | None = None
    desde_experimento: str | None = None
    desde_barrido: str | None = None
    criterio: Literal["mejor", "meseta", "elegida"] = "mejor"
    id_experimento: str | None = None
    manual: RecetaManual | None = None


class PeticionVersion(BaseModel):
    cambios: dict[str, Any]


class PeticionBacktest(BaseModel):
    fecha_inicio: str | None = None
    fecha_fin: str | None = None
    version: int | None = None
    capital_inicial: float | None = Field(None, gt=0)
    forzar: bool = False


class PeticionComparativaConfiguracion(PeticionComparativa):
    """Estrategias a comparar (como en un marco) más las fechas y la versión de la configuración (opcionales)."""

    fecha_inicio: str | None = None
    fecha_fin: str | None = None
    version: int | None = None


def _configuracion(ctx: Contexto, id_configuracion: str, version: int | None = None) -> ConfiguracionGuardada:
    try:
        return ctx.almacen_configuraciones.leer(id_configuracion, version)
    except (KeyError, ValueError) as error:
        raise HTTPException(404, f"Configuración '{id_configuracion}'{f' v{version}' if version else ''} no encontrada") from error


def _metricas_planas(ctx: Contexto, id_experimento: str) -> dict[str, Any]:
    """Métricas numéricas de nivel superior de un experimento (las que se guardan como origen)."""
    metricas = ctx.almacen.leer_metricas(id_experimento) or {}
    return {k: v for k, v in metricas.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _config_de_experimento(ctx: Contexto, id_experimento: str) -> ConfiguracionExperimento:
    """Configuración de un experimento TERMINADO (404 si no existe, 409 si no ha terminado, 422 si no es relanzable)."""
    try:
        estado = ctx.almacen.estado_de(id_experimento)
    except ValueError:
        estado = None
    if estado is None:
        raise HTTPException(404, f"Experimento '{id_experimento}' no encontrado")
    if estado != "hecho":
        raise HTTPException(409, f"El experimento '{id_experimento}' no ha terminado (estado: {estado})")
    try:
        return ConfiguracionExperimento(**ctx.almacen.leer_config(id_experimento))
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error


def _origen_experimento(ctx: Contexto, id_experimento: str, config: ConfiguracionExperimento) -> OrigenConfiguracion:
    return OrigenConfiguracion(tipo="experimento", id_experimento=id_experimento, criterio="elegida",
                               metricas=_metricas_planas(ctx, id_experimento), fechas=[config.fecha_inicio, config.fecha_fin])


def _origen_barrido(ctx: Contexto, peticion: PeticionConfiguracion) -> tuple[ConfiguracionExperimento, OrigenConfiguracion]:
    """Elige la combinación del barrido según el criterio y devuelve su configuración y el origen."""
    id_barrido = peticion.desde_barrido
    try:
        existe = ctx.almacen_barridos.existe(id_barrido)
    except ValueError:
        existe = False
    if not existe:
        raise HTTPException(404, f"Barrido '{id_barrido}' no encontrado")
    barrido = ctx.almacen_barridos.cargar(id_barrido)
    filas = [f for f in superficie_registros(barrido) if f["objetivo"] is not None]
    if not filas:
        raise HTTPException(409, "El barrido no tiene ninguna combinación terminada")
    if peticion.criterio == "elegida":
        if peticion.id_experimento not in {f["id"] for f in filas}:
            raise HTTPException(422, f"'{peticion.id_experimento}' no es una combinación terminada del barrido {id_barrido}")
        fila = next(f for f in filas if f["id"] == peticion.id_experimento)
    elif peticion.criterio == "meseta":
        con_meseta = [f for f in filas if f["meseta"] is not None]
        if not con_meseta:
            raise HTTPException(422, "Ninguna combinación del barrido tiene vecinos: no hay meseta que calcular")
        fila = max(con_meseta, key=lambda f: f["meseta"])
    else:
        fila = max(filas, key=lambda f: f["objetivo"])
    config = _config_de_experimento(ctx, fila["id"])
    avisos = [AVISO_BARRIDO]
    if barrido.config.fecha_fin_desarrollo and ctx.almacen_barridos.leer_resultado(id_barrido, "reservado") is None:
        avisos.append(f"Sin validar en el periodo reservado (desde {barrido.config.fecha_fin_desarrollo}).")
    metricas = {**fila["metricas"], "meseta": fila["meseta"], "n_vecinos": fila["n_vecinos"], "objetivo": fila["objetivo"],
                "metrica_objetivo": barrido.config.metrica_objetivo}
    return config, OrigenConfiguracion(tipo="barrido", id_barrido=id_barrido, id_experimento=fila["id"], criterio=peticion.criterio,
                                       metricas=metricas, fechas=[config.fecha_inicio, config.fecha_fin], aviso=" ".join(avisos))


@router.get("")
def listar_configuraciones(estrategia: str | None = None, texto: str | None = None, archivadas: bool = False,
                           ctx: Contexto = Depends(obtener_contexto)):
    return {"configuraciones": ctx.almacen_configuraciones.listar(estrategia=estrategia, texto=texto, archivadas=archivadas)}


@router.post("", status_code=201)
def crear_configuracion(peticion: PeticionConfiguracion, ctx: Contexto = Depends(obtener_contexto)):
    fuentes = [f for f in (peticion.desde_experimento, peticion.desde_barrido, peticion.manual) if f is not None]
    if len(fuentes) != 1:
        raise HTTPException(422, "Indica exactamente una fuente: desde_experimento, desde_barrido o manual")
    if peticion.desde_experimento is not None:
        config = _config_de_experimento(ctx, peticion.desde_experimento)
        origen = _origen_experimento(ctx, peticion.desde_experimento, config)
    elif peticion.desde_barrido is not None:
        config, origen = _origen_barrido(ctx, peticion)
    else:
        receta = peticion.manual
        try:
            config = ConfiguracionExperimento(**receta.model_dump(), fecha_inicio="2000-01-01", fecha_fin="2000-01-02")
        except ValidationError as error:
            raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error
        origen = OrigenConfiguracion(tipo="manual")
    if peticion.id_marco is not None:
        try:
            ctx.almacen_marcos.leer(peticion.id_marco)
        except (KeyError, ValueError) as error:
            raise HTTPException(404, f"Marco '{peticion.id_marco}' no encontrado") from error
    try:
        configuracion = ConfiguracionGuardada.desde_experimento(
            config, nombre=peticion.nombre, notas=peticion.notas, origen=origen, id_marco=peticion.id_marco,
            etiquetas=peticion.etiquetas, id=peticion.id)
        return ctx.almacen_configuraciones.guardar(configuracion)
    except KeyError as error:
        raise HTTPException(404, str(error.args[0])) from error
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error
    except ConfiguracionExistente as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/{id_configuracion}")
def detalle_configuracion(id_configuracion: str, version: int | None = None, ctx: Contexto = Depends(obtener_contexto)):
    configuracion = _configuracion(ctx, id_configuracion, version)
    versiones = [{"version": v.version, "creada": v.creada, "deriva_de": v.deriva_de, "notas": v.notas}
                 for v in ctx.almacen_configuraciones.versiones(id_configuracion)]
    marco = ctx.servicio.marco_enlazado(configuracion)  # None si no tiene o se borró: la configuración sigue siendo válida
    comparativas = [con_estado(ctx, c) for c in ctx.almacen_marcos.listar_comparativas()
                    if c["marco"].get("id_configuracion") == id_configuracion]
    return {"configuracion": configuracion, "versiones": versiones, "marco": marco,
            "backtests": ctx.servicio.backtests_de_receta(configuracion),
            "fechas_sugeridas": list(fechas_por_defecto(configuracion, marco)), "comparativas": comparativas}


@router.post("/{id_configuracion}/versiones", status_code=201)
def crear_version(id_configuracion: str, peticion: PeticionVersion, ctx: Contexto = Depends(obtener_contexto)):
    _configuracion(ctx, id_configuracion)
    try:
        return ctx.almacen_configuraciones.nueva_version(id_configuracion, peticion.cambios)
    except KeyError as error:
        raise HTTPException(404, str(error.args[0])) from error
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/{id_configuracion}/backtest", status_code=202)
def lanzar_backtest(id_configuracion: str, peticion: PeticionBacktest, ctx: Contexto = Depends(obtener_contexto)):
    configuracion = _configuracion(ctx, id_configuracion, peticion.version)
    try:
        id_experimento, reutilizado = ctx.servicio.lanzar_backtest_de_configuracion(
            configuracion, peticion.fecha_inicio, peticion.fecha_fin, peticion.capital_inicial, peticion.forzar)
    except KeyError as error:
        raise HTTPException(404, str(error.args[0])) from error
    except (ValidationError, ValueError) as error:
        detalle = error.errors(include_url=False, include_context=False) if isinstance(error, ValidationError) else str(error)
        raise HTTPException(422, detalle) from error
    return {"id_experimento": id_experimento, "reutilizado": reutilizado}


@router.post("/{id_configuracion}/comparativa", status_code=202)
def lanzar_comparativa_de_configuracion(id_configuracion: str, peticion: PeticionComparativaConfiguracion,
                                        ctx: Contexto = Depends(obtener_contexto)):
    configuracion = _configuracion(ctx, id_configuracion, peticion.version)
    try:
        return ctx.servicio.lanzar_comparativa_de_configuracion(
            configuracion, [e.model_dump() for e in peticion.estrategias], peticion.fecha_inicio, peticion.fecha_fin,
            peticion.forzar)
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False, include_input=False)) from error
    except ValueError as error:  # marco incoherente o estrategias que no validan (todas se validan antes de lanzar)
        raise HTTPException(422, str(error)) from error


@router.delete("/{id_configuracion}", status_code=204)
def archivar_configuracion(id_configuracion: str, definitivo: bool = False, ctx: Contexto = Depends(obtener_contexto)):
    _configuracion(ctx, id_configuracion)
    if definitivo:
        ctx.almacen_configuraciones.borrar(id_configuracion)
    else:
        ctx.almacen_configuraciones.archivar(id_configuracion)
