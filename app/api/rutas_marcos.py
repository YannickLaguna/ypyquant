"""
Rutas de marcos de comparación y de las comparativas que se lanzan sobre ellos.

Responsabilidad:
- `GET/POST /api/marcos`, `GET/PUT/DELETE /api/marcos/{id}`: el entorno común (universo, fechas, frecuencia,
  costes, benchmark, capital, semilla, historia previa y parámetros comunes). `GET /api/marcos/{id}` devuelve el
  marco y sus comparativas. 409 si el id ya existe; 422 si el marco no valida.
- `POST /api/marcos/{id}/comparativa` (202): `{estrategias: [{estrategia, parametros}], forzar?}` aplica el marco a cada
  estrategia (`aplicar_marco`), valida TODAS antes de lanzar nada (422 con los fallos de todas), crea N experimentos
  normales (los ya calculados se reutilizan: misma huella) y guarda el registro de la comparativa con el informe de
  qué comunes se aplicaron a cada estrategia y los avisos.
- `GET /api/comparativas` (filtro `id_marco`) y `GET /api/comparativas/{id}`: el registro con el estado actual de
  cada experimento (para abrir la página Comparar con todos de golpe).

Se relaciona con: app/api/contexto.py (almacenes y `lanzar_experimento`), ypyquant.estrategias.marcos (toda la lógica),
app/api/rutas_experimentos.py (`GET /api/comparar` avisa si no es ceteris paribus) y app/web (páginas Marcos y Comparar).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.api.contexto import Contexto, lanzar_experimento, obtener_contexto
from ypyquant.estrategias.marcos import MarcoComparacion, MarcoExistente, preparar_comparativa

router = APIRouter(prefix="/api", tags=["marcos"])


class EntradaEstrategia(BaseModel):
    """Una estrategia de la comparativa con sus parámetros PROPIOS (los comunes los pone el marco)."""

    estrategia: str
    parametros: dict[str, Any] = Field(default_factory=dict)


class PeticionComparativa(BaseModel):
    estrategias: list[EntradaEstrategia] = Field(min_length=1, max_length=20)
    forzar: bool = False


def _marco(ctx: Contexto, id_marco: str) -> MarcoComparacion:
    try:
        return ctx.almacen_marcos.leer(id_marco)
    except (KeyError, ValueError) as error:
        raise HTTPException(404, f"Marco '{id_marco}' no encontrado") from error


def _errores_validacion(error: ValidationError) -> list[dict[str, Any]]:
    return error.errors(include_url=False, include_context=False)


@router.get("/marcos")
def listar_marcos(ctx: Contexto = Depends(obtener_contexto)):
    return {"marcos": ctx.almacen_marcos.listar()}


@router.post("/marcos", status_code=201)
def crear_marco(marco: MarcoComparacion, ctx: Contexto = Depends(obtener_contexto)):
    try:
        return ctx.almacen_marcos.guardar(marco)
    except MarcoExistente as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/marcos/{id_marco}")
def detalle_marco(id_marco: str, ctx: Contexto = Depends(obtener_contexto)):
    return {"marco": _marco(ctx, id_marco), "comparativas": ctx.almacen_marcos.listar_comparativas(id_marco)}


@router.put("/marcos/{id_marco}")
def actualizar_marco(id_marco: str, marco: MarcoComparacion, ctx: Contexto = Depends(obtener_contexto)):
    """Reemplaza el marco (el id de la URL manda). Los experimentos ya lanzados no cambian: son recetas congeladas."""
    _marco(ctx, id_marco)
    try:
        return ctx.almacen_marcos.actualizar(marco.model_copy(update={"id": id_marco}))
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.delete("/marcos/{id_marco}", status_code=204)
def borrar_marco(id_marco: str, ctx: Contexto = Depends(obtener_contexto)):
    try:
        existia = ctx.almacen_marcos.borrar(id_marco)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    if not existia:
        raise HTTPException(404, f"Marco '{id_marco}' no encontrado")


def lanzar_sobre_marco(ctx: Contexto, marco: MarcoComparacion, peticion: PeticionComparativa) -> dict[str, Any]:
    """
    Aplica `marco` a cada estrategia de la petición validándolas TODAS antes de lanzar nada (422 con los fallos),
    crea los experimentos normales (los ya calculados se reutilizan) y registra la comparativa. Lo comparten
    `POST /api/marcos/{id}/comparativa` y `POST /api/configuraciones/{id}/comparativa`.
    """
    try:
        preparadas = preparar_comparativa(marco, [e.model_dump() for e in peticion.estrategias])
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    entradas = []
    for entrada, (config, informe) in zip(peticion.estrategias, preparadas, strict=True):
        id_experimento, reutilizado = lanzar_experimento(ctx, config, peticion.forzar)
        entradas.append({"id_experimento": id_experimento, "estrategia": entrada.estrategia,
                         "parametros": entrada.parametros, "informe": informe.model_dump(), "reutilizado": reutilizado})
    return ctx.almacen_marcos.registrar_comparativa(marco, entradas)


@router.post("/marcos/{id_marco}/comparativa", status_code=202)
def lanzar_comparativa(id_marco: str, peticion: PeticionComparativa, ctx: Contexto = Depends(obtener_contexto)):
    return lanzar_sobre_marco(ctx, _marco(ctx, id_marco), peticion)


def con_estado(ctx: Contexto, comparativa: dict[str, Any]) -> dict[str, Any]:
    """La comparativa guardada con el estado actual de cada experimento (None si se borró)."""
    experimentos = []
    for e in comparativa["experimentos"]:
        try:
            estado = ctx.almacen.estado_de(e["id_experimento"])
        except ValueError:
            estado = None
        experimentos.append({**e, "estado": estado})
    return {**comparativa, "experimentos": experimentos}


@router.get("/comparativas")
def listar_comparativas(id_marco: str | None = None, ctx: Contexto = Depends(obtener_contexto)):
    return {"comparativas": [con_estado(ctx, c) for c in ctx.almacen_marcos.listar_comparativas(id_marco)]}


@router.get("/comparativas/{id_comparativa}")
def detalle_comparativa(id_comparativa: str, ctx: Contexto = Depends(obtener_contexto)):
    try:
        return con_estado(ctx, ctx.almacen_marcos.leer_comparativa(id_comparativa))
    except KeyError as error:
        raise HTTPException(404, f"Comparativa '{id_comparativa}' no encontrada") from error
