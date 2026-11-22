"""
Rutas de barridos (rejillas de experimentos) y su validación estadística.

Responsabilidad:
- `POST /api/barridos`: crea el barrido y encola todas sus combinaciones (202). `GET /api/barridos`
  (lista), `GET /api/barridos/{id}` (configuración, estado, progreso, tabla de métricas con la
  meseta, tareas y resultados de validación disponibles), `DELETE /api/barridos/{id}`
  (`?con_experimentos=true` borra también los experimentos).
- Validación (todas exigen el barrido terminado; 409 si quedan experimentos pendientes o en curso):
  `POST .../validar` (DSR, PBO, rachas, meseta), `POST .../walk_forward`, `POST .../reservado`
  (síncronas: tardan segundos) y `POST .../permutacion` (202, en segundo plano con progreso en
  `tareas.permutacion_<modo>`; `POST .../permutacion/cancelar` la detiene).
- Contratos JSON de cada ruta: tabla en docs/ui.md.

Códigos: 404 barrido o estrategia inexistente, 409 experimentos sin terminar o tarea ya en curso,
422 parámetros inválidos (mensaje en `detail`).

Se relaciona con: app/api/contexto.py (almacén, ejecutores y `ctx.servicio`, que lanza y cancela las permutaciones), ypyquant.estrategias.barridos,
.validacion_barridos y .reservado (toda la lógica; esta capa solo traduce a HTTP).
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.api.contexto import Contexto, obtener_contexto
from ypyquant.estrategias.barridos import Barrido, ConfiguracionBarrido, superficie_registros
from ypyquant.estrategias.reservado import RegistroReservado, clave_reservado, evaluar_reservado
from ypyquant.estrategias.validacion_barridos import (
    BarridoEnCurso,
    exigir_terminado,
    validar_barrido,
    validar_parametros_permutacion,
    walk_forward_barrido,
)

router = APIRouter(prefix="/api", tags=["barridos"])

RESULTADOS = ("validacion", "walk_forward", "permutacion_in_sample", "permutacion_walk_forward")


class PeticionValidar(BaseModel):
    """`n_bloques`: bloques del PBO-CSCV (se reduce solo si hay pocos datos); `relleno`: ver matriz de retornos."""

    n_bloques: int = Field(16, ge=4, le=64)
    semilla: int = 0
    relleno: Literal["recortar", "cero"] = "recortar"


class PeticionPermutacion(BaseModel):
    """Parámetros de la prueba de permutación (límites: 1..1000 permutaciones, procesos 1..32)."""

    n_permutaciones: int = Field(100, ge=1)
    longitud_bloque: int = Field(1, ge=1)
    modo: Literal["in_sample", "walk_forward"] = "in_sample"
    ventana_entreno_dias: int | None = Field(None, ge=1)
    paso_dias: int | None = Field(None, ge=1)
    semilla: int = 0
    procesos: int = Field(1, ge=1, le=32)


class PeticionWalkForward(BaseModel):
    ventana_entreno_dias: int = Field(..., ge=1)
    paso_dias: int = Field(..., ge=1)


class PeticionReservado(BaseModel):
    """`experimento`: id de experimento o etiqueta (`top_k=2, meses_retorno=6`) de una combinación del barrido."""

    experimento: str


class PeticionCancelar(BaseModel):
    modo: Literal["in_sample", "walk_forward"] = "in_sample"


def _cargar(ctx: Contexto, id_barrido: str) -> Barrido:
    try:
        return ctx.almacen_barridos.cargar(id_barrido)
    except (KeyError, ValueError) as error:
        raise HTTPException(404, f"Barrido '{id_barrido}' no encontrado") from error


def _terminado(barrido: Barrido, minimo: int = 1) -> None:
    """409 si hay experimentos sin terminar; 422 si no hay suficientes combinaciones terminadas bien."""
    try:
        exigir_terminado(barrido, minimo)
    except BarridoEnCurso as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


def _ejecutar(funcion, *args, **kwargs):
    """Llama a la lógica de validación traduciendo sus errores: BarridoEnCurso -> 409, ValueError -> 422."""
    try:
        return funcion(*args, **kwargs)
    except BarridoEnCurso as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except RuntimeError as error:  # p. ej. sin precios para el universo
        raise HTTPException(422, str(error)) from error


@router.post("/barridos", status_code=202)
def crear_barrido(config: ConfiguracionBarrido, forzar: bool = False, ctx: Contexto = Depends(obtener_contexto)):
    try:
        config.base.normalizada()
        combinaciones = config.combinaciones()
    except KeyError as error:
        raise HTTPException(404, str(error.args[0])) from error
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    omitidas = config.omitidas()
    if not combinaciones:
        motivos = "; ".join(dict.fromkeys(o["motivo"] for o in omitidas))
        raise HTTPException(422, f"Ninguna combinación de la rejilla es válida ({motivos})")
    id_barrido, ids, reutilizado = ctx.servicio.lanzar_barrido(config, forzar)
    return {"id": id_barrido, "ids": ids, "total": len(combinaciones), "reutilizado": reutilizado,
            "omitidas": omitidas}


@router.get("/barridos")
def listar_barridos(ctx: Contexto = Depends(obtener_contexto)):
    return {"barridos": ctx.almacen_barridos.listar()}


@router.get("/barridos/{id_barrido}")
def detalle_barrido(id_barrido: str, ctx: Contexto = Depends(obtener_contexto)):
    barrido = _cargar(ctx, id_barrido)
    almacen = ctx.almacen_barridos
    datos = almacen.leer(id_barrido)
    combinaciones = [{**c, "estado": ctx.almacen.estado_de(c["id"]) or "error"} for c in datos["combinaciones"]]
    config = barrido.config
    usos = []
    if config.fecha_fin_desarrollo:
        clave = clave_reservado(config.base.universo, config.fecha_fin_desarrollo, config.base.fecha_fin)
        usos = RegistroReservado(almacen.ruta_reservado).usos(clave)
    return {
        "id": id_barrido, "config": datos["config"], "estado": datos["estado"], "progreso": datos["progreso"],
        "creado": datos["creado"], "actualizado": datos["actualizado"], "ejes": config.rejilla,
        "combinaciones": combinaciones, "omitidas": datos.get("omitidas", []), "superficie": superficie_registros(barrido) if datos["estado"] != "pendiente" else [],
        "tareas": datos["tareas"],
        **{nombre: almacen.leer_resultado(id_barrido, nombre) for nombre in RESULTADOS},
        "reservado": almacen.leer_resultado(id_barrido, "reservado") or [],
        "reservado_usos": usos,
        "periodo_reservado": ({"inicio": (pd.Timestamp(config.fecha_fin_desarrollo) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                               "fin": config.base.fecha_fin} if config.fecha_fin_desarrollo else None),
    }


@router.delete("/barridos/{id_barrido}", status_code=204)
def borrar_barrido(id_barrido: str, con_experimentos: bool = False, ctx: Contexto = Depends(obtener_contexto)):
    try:
        existia = ctx.almacen_barridos.borrar(id_barrido, con_experimentos)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    if not existia:
        raise HTTPException(404, f"Barrido '{id_barrido}' no encontrado")


@router.post("/barridos/{id_barrido}/validar")
def validar(id_barrido: str, peticion: PeticionValidar | None = None, ctx: Contexto = Depends(obtener_contexto)):
    peticion = peticion or PeticionValidar()
    return _ejecutar(validar_barrido, _cargar(ctx, id_barrido), n_bloques=peticion.n_bloques, semilla=peticion.semilla,
                     relleno=peticion.relleno)


@router.post("/barridos/{id_barrido}/walk_forward")
def walk_forward(id_barrido: str, peticion: PeticionWalkForward, ctx: Contexto = Depends(obtener_contexto)):
    return _ejecutar(walk_forward_barrido, _cargar(ctx, id_barrido), peticion.ventana_entreno_dias, peticion.paso_dias)


@router.post("/barridos/{id_barrido}/permutacion", status_code=202)
def permutacion(id_barrido: str, peticion: PeticionPermutacion, ctx: Contexto = Depends(obtener_contexto)):
    barrido = _cargar(ctx, id_barrido)
    parametros: dict[str, Any] = peticion.model_dump()
    try:
        validar_parametros_permutacion(peticion.n_permutaciones, peticion.longitud_bloque, peticion.modo,
                                       peticion.ventana_entreno_dias, peticion.paso_dias, peticion.procesos)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    _terminado(barrido, minimo=2 if peticion.modo == "walk_forward" else 1)
    try:
        tarea = ctx.servicio.lanzar_permutacion(id_barrido, parametros)
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error
    n_backtests = len(barrido.ids) * peticion.n_permutaciones
    return {"tarea": tarea, "estado": "pendiente", "n_backtests": n_backtests,
            "aviso": "Más de 20000 backtests: puede tardar horas" if n_backtests > 20_000 else None}


@router.post("/barridos/{id_barrido}/permutacion/cancelar")
def cancelar_permutacion(id_barrido: str, peticion: PeticionCancelar | None = None,
                         ctx: Contexto = Depends(obtener_contexto)):
    _cargar(ctx, id_barrido)
    nombre = ctx.servicio.nombre_tarea_permutacion((peticion or PeticionCancelar()).modo)
    if not ctx.servicio.cancelar_permutacion(id_barrido, nombre):
        raise HTTPException(409, f"No hay una {nombre} en curso")
    return {"tarea": nombre, "cancelando": True}


@router.post("/barridos/{id_barrido}/reservado")
def reservado(id_barrido: str, peticion: PeticionReservado, ctx: Contexto = Depends(obtener_contexto)):
    barrido = _cargar(ctx, id_barrido)
    if barrido.config.fecha_fin_desarrollo is None:
        raise HTTPException(422, "Este barrido no tiene periodo reservado (falta fecha_fin_desarrollo)")
    _terminado(barrido)
    return _ejecutar(evaluar_reservado, barrido, peticion.experimento, ctx.cargador_precios)
