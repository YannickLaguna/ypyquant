"""
Rutas de experimentos (backtests) y comparación.

Responsabilidad:
- `POST /api/experimentos`: valida la configuración y la encola en segundo plano (202). Misma
  configuración = mismo id: se reutiliza el resultado salvo `?forzar=true`.
- `POST /api/experimentos/barrido`: expande una rejilla de parámetros y encola N experimentos;
  además crea el barrido equivalente (ver rutas_barridos.py) y devuelve su `barrido_id` (None si no
  hay rejilla, es decir, con una sola combinación).
- `GET /api/validadores`: nombres de los validadores por experimento (para `config.validaciones`).
- `GET /api/experimentos` (filtros `estrategia`, `estado`, `texto`), `GET /api/experimentos/{id}`
  (métricas, equity, drawdown, pesos y operaciones), `DELETE /api/experimentos/{id}`.
- `GET /api/comparar?ids=a,b,c`: equity normalizada a base 100 y métricas lado a lado (DuckDB).

Se relaciona con: app/api/contexto.py (almacén y ejecutor) y ypyquant.estrategias.experimentos
(ConfiguracionExperimento, expandir_rejilla). Los errores de validación de la estrategia (nombre o
parámetros inválidos) se devuelven como 422/404 antes de encolar nada.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ValidationError

from app.api.contexto import Contexto, lanzar_barrido, lanzar_experimento, obtener_contexto
from ypyquant.estrategias.barridos import ConfiguracionBarrido
from ypyquant.estrategias.experimentos import VALIDADORES, ConfiguracionExperimento, expandir_rejilla

router = APIRouter(prefix="/api", tags=["experimentos"])


class PeticionBarrido(BaseModel):
    """Configuración base + `{parametro_o_campo: [valores]}` a combinar."""

    config: ConfiguracionExperimento
    rejilla: dict[str, list[Any]]


def _validar(config: ConfiguracionExperimento) -> ConfiguracionExperimento:
    """Normaliza la configuración traduciendo los fallos a códigos HTTP (404 estrategia, 422 parámetros)."""
    try:
        return config.normalizada()
    except KeyError as error:
        raise HTTPException(404, str(error.args[0])) from error
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error


@router.post("/experimentos", status_code=202)
def crear_experimento(config: ConfiguracionExperimento, forzar: bool = False, ctx: Contexto = Depends(obtener_contexto)):
    id_experimento, reutilizado = lanzar_experimento(ctx, _validar(config), forzar)
    return {"id": id_experimento, "reutilizado": reutilizado}


@router.post("/experimentos/barrido", status_code=202)
def crear_barrido(peticion: PeticionBarrido, forzar: bool = False, ctx: Contexto = Depends(obtener_contexto)):
    base = _validar(peticion.config)
    try:
        configuraciones = expandir_rejilla(base, peticion.rejilla)
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if peticion.rejilla:
        try:
            barrido = ConfiguracionBarrido(base=base, rejilla=peticion.rejilla)
        except ValueError as error:  # p. ej. un eje con lista vacía
            raise HTTPException(422, str(error)) from error
        id_barrido, ids, _ = lanzar_barrido(ctx, barrido, forzar)
        return {"ids": ids, "total": len(ids), "barrido_id": id_barrido}
    ids = [lanzar_experimento(ctx, c, forzar)[0] for c in configuraciones]
    return {"ids": ids, "total": len(ids), "barrido_id": None}


@router.get("/validadores")
def listar_validadores():
    return {"validadores": sorted(VALIDADORES)}


@router.get("/experimentos")
def listar_experimentos(estrategia: str | None = None, estado: str | None = None, texto: str | None = None,
                        ctx: Contexto = Depends(obtener_contexto)):
    return {"experimentos": ctx.almacen.listar(estrategia=estrategia, estado=estado, texto=texto)}


@router.get("/experimentos/{id_experimento}")
def detalle_experimento(id_experimento: str, ctx: Contexto = Depends(obtener_contexto)):
    try:
        return ctx.almacen.leer(id_experimento)
    except (KeyError, ValueError) as error:
        raise HTTPException(404, f"Experimento '{id_experimento}' no encontrado") from error


@router.delete("/experimentos/{id_experimento}", status_code=204)
def borrar_experimento(id_experimento: str, ctx: Contexto = Depends(obtener_contexto)):
    try:
        existia = ctx.almacen.borrar(id_experimento)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    if not existia:
        raise HTTPException(404, f"Experimento '{id_experimento}' no encontrado")


@router.get("/comparar")
def comparar(ids: str = Query(..., description="Identificadores separados por comas"),
             ctx: Contexto = Depends(obtener_contexto)):
    lista = [i.strip() for i in ids.split(",") if i.strip()]
    if len(lista) < 1:
        raise HTTPException(422, "Indica al menos un id")
    try:
        return ctx.almacen.comparar(lista)
    except (KeyError, ValueError) as error:
        raise HTTPException(404, str(error.args[0])) from error
