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
  (métricas, equity, drawdown, pesos y operaciones), `DELETE /api/experimentos/{id}` y
  `POST /api/experimentos/borrar` (`{"ids": [...]}`: borrado múltiple; omite los que siguen en marcha).
- `GET /api/comparar?ids=a,b,c`: equity normalizada a base 100 y métricas lado a lado (DuckDB), más
  `ceteris_paribus: {ok, diferencias}` (en qué difieren del entorno o de los parámetros comunes; ver `marcos.py`).
- `GET /api/comparar/series?ids=a,b&indicador=`: series temporales por experimento para la página Comparar
  (`drawdown`, `sharpe_movil`, `volatilidad_movil` sobre el eje de fechas común de `/api/comparar`, y
  `turnover_rebalanceo` con los rebalanceos propios de cada uno recortados a ese rango). Responde
  `{indicador, ids, series: {id: {fechas, valores}}, sin_dato: [ids]}`; un experimento sin pesos (TFM importado) o con
  serie demasiado corta va en `sin_dato`, nunca como ceros. Se calcula en el servidor (no en el cliente) porque la
  periodicidad (mensual del TFM frente a diaria) decide la ventana de 12 meses y las funciones viven en
  `ypyquant.estrategias.series_comparacion`. 422 si el indicador no existe.

Se relaciona con: app/api/contexto.py (almacén y `ctx.servicio`) y ypyquant.estrategias (`ServicioExperimentos`,
ConfiguracionExperimento, expandir_rejilla). Los errores de validación de la estrategia (nombre o
parámetros inválidos) se devuelven como 422/404 antes de encolar nada.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

from app.api.contexto import Contexto, obtener_contexto
from ypyquant.estrategias.barridos import ConfiguracionBarrido
from ypyquant.estrategias.experimentos import VALIDADORES, ConfiguracionExperimento, expandir_rejilla_con_omitidas
from ypyquant.estrategias.marcos import diferencias_ceteris_paribus
from ypyquant.estrategias.series_comparacion import (
    INDICADORES_TEMPORALES,
    drawdown_serie,
    periodos_por_anio,
    sharpe_movil,
    turnover_por_rebalanceo,
    volatilidad_movil,
)

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
    id_experimento, reutilizado = ctx.servicio.lanzar(_validar(config), forzar)
    return {"id": id_experimento, "reutilizado": reutilizado}


@router.post("/experimentos/barrido", status_code=202)
def crear_barrido(peticion: PeticionBarrido, forzar: bool = False, ctx: Contexto = Depends(obtener_contexto)):
    base = _validar(peticion.config)
    try:
        configuraciones, omitidas = expandir_rejilla_con_omitidas(base, peticion.rejilla)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if not configuraciones:
        motivos = "; ".join(dict.fromkeys(o["motivo"] for o in omitidas))
        raise HTTPException(422, f"Ninguna combinación de la rejilla es válida ({motivos})")
    if peticion.rejilla:
        try:
            barrido = ConfiguracionBarrido(base=base, rejilla=peticion.rejilla)
        except ValueError as error:  # p. ej. un eje con lista vacía
            raise HTTPException(422, str(error)) from error
        id_barrido, ids, _ = ctx.servicio.lanzar_barrido(barrido, forzar)
        return {"ids": ids, "total": len(ids), "barrido_id": id_barrido, "omitidas": omitidas}
    ids = ctx.servicio.lanzar_varios(configuraciones, forzar)
    return {"ids": ids, "total": len(ids), "barrido_id": None, "omitidas": omitidas}


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


class PeticionBorrarVarios(BaseModel):
    """Lista de ids a borrar de una vez (selección múltiple de la interfaz)."""

    ids: list[str] = Field(min_length=1, max_length=5000)


@router.post("/experimentos/borrar")
def borrar_varios_experimentos(peticion: PeticionBorrarVarios, ctx: Contexto = Depends(obtener_contexto)):
    """
    Borra varios experimentos y devuelve qué pasó con cada id: `borrados`, `en_marcha` (pendientes o en
    curso: no se tocan porque el ejecutor seguiría escribiendo en su carpeta) y `no_encontrados`.
    """
    return ctx.servicio.borrar_varios(peticion.ids)


@router.get("/comparar")
def comparar(ids: str = Query(..., description="Identificadores separados por comas"),
             ctx: Contexto = Depends(obtener_contexto)):
    lista = [i.strip() for i in ids.split(",") if i.strip()]
    if len(lista) < 1:
        raise HTTPException(422, "Indica al menos un id")
    try:
        comparacion = ctx.almacen.comparar(lista)
    except (KeyError, ValueError) as error:
        raise HTTPException(404, str(error.args[0])) from error
    configs = {i: ctx.almacen.leer_config(i) or {} for i in comparacion["ids"]}
    diferencias = diferencias_ceteris_paribus(configs)
    return {**comparacion, "ceteris_paribus": {"ok": not diferencias, "diferencias": diferencias}}


@router.get("/comparar/series")
def comparar_series(ids: str = Query(..., description="Identificadores separados por comas"),
                    indicador: str = Query(..., description=f"Uno de: {', '.join(INDICADORES_TEMPORALES)}"),
                    ctx: Contexto = Depends(obtener_contexto)):
    """Serie temporal de `indicador` por experimento (ver la cabecera del módulo y `series_comparacion.py`)."""
    if indicador not in INDICADORES_TEMPORALES:
        raise HTTPException(422, f"Indicador desconocido '{indicador}' (disponibles: {', '.join(INDICADORES_TEMPORALES)})")
    lista = [i.strip() for i in ids.split(",") if i.strip()]
    if not lista:
        raise HTTPException(422, "Indica al menos un id")
    try:
        comparacion = ctx.almacen.comparar(lista)
    except (KeyError, ValueError) as error:
        raise HTTPException(404, str(error.args[0])) from error
    fechas = comparacion["fechas"]
    periodos = periodos_por_anio(fechas)
    series: dict[str, dict[str, list]] = {}
    for id_experimento in comparacion["ids"]:
        if indicador == "turnover_rebalanceo":
            pesos = ctx.almacen.leer_pesos(id_experimento)
            fechas_t, valores = turnover_por_rebalanceo(pesos) if pesos is not None else ([], [])
            recortadas = [(f, v) for f, v in zip(fechas_t, valores, strict=True) if fechas[0] <= f <= fechas[-1]]
            series[id_experimento] = {"fechas": [f for f, _ in recortadas], "valores": [v for _, v in recortadas]}
            continue
        equity = comparacion["series"][id_experimento]
        valores = {"drawdown": lambda: drawdown_serie(equity),
                   "sharpe_movil": lambda: sharpe_movil(equity, periodos),
                   "volatilidad_movil": lambda: volatilidad_movil(equity, periodos)}[indicador]()
        series[id_experimento] = {"fechas": fechas, "valores": valores}
    sin_dato = [i for i, s in series.items() if not any(v is not None for v in s["valores"])]
    return {"indicador": indicador, "etiqueta": INDICADORES_TEMPORALES[indicador], "ids": comparacion["ids"],
            "periodos_por_anio": periodos, "series": series, "sin_dato": sin_dato}
