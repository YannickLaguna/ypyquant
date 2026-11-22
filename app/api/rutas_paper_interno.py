"""
Rutas del paper interno: sesiones forward de una receta congelada, con precios reales y sin broker.

Responsabilidad (prefijo `/api/paper/interno`; distinto de `/api/paper`, que es el paper de Alpaca):
- `GET /sesiones`: una fila por sesión (estado, retorno, Sharpe, drawdown, costes, días, `dias_pendientes`, `al_dia`).
- `POST /sesiones`: `{id_experimento | id_configuracion (+version), capital_inicial?, modo_ejecucion, comision?,
  slippage?, nombre?, fecha_alta?}` crea una sesión desde un experimento TERMINADO (o una configuración guardada):
  congela su receta y deja la cartera en efectivo. 201 con el detalle. `modo_ejecucion`: `apertura_siguiente` (por
  defecto: decide con el cierre de D, ejecuta a la apertura de D+1) o `cierre` (como el backtest, optimista).
- `GET /sesiones/{id}`: detalle con receta, métricas forward (las de `evaluar`), posiciones, pesos objetivo y avisos;
  `GET /sesiones/{id}/equity|operaciones|pesos`: series y tablas.
- `POST /sesiones/{id}/pausar|reanudar|detener`: control (una sesión detenida no se reanuda: 409).
- `POST /avanzar` (202): avanza todas las sesiones activas en un hilo del ejecutor; 409 si ya hay un avance en curso.
  `GET /avanzar/estado`: `{en_curso, inicio, fin, resultados, error}` para el sondeo de la web.
- `GET /sesiones/{id}/espejo`: lanza (o reutiliza por huella) el backtest de la misma receta de `fecha_alta` a la última
  fecha procesada y devuelve la curva real y la del backtest para compararlas (`estado` del espejo mientras se calcula).

Códigos: 404 sesión, experimento o configuración inexistente; 409 experimento sin terminar, avance en curso o
transición inválida; 422 petición incoherente (fuentes, estrategia sin código, universo dinámico, modo inválido).

Se relaciona con: app/api/contexto.py (`servicio_paper_interno`, `control_avance_paper`, `ctx.servicio` para el
espejo), ypyquant.paper (ServicioPaper) y app/web (página «Paper interno»).
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.api.contexto import Contexto, obtener_contexto
from ypyquant.paper.servicio import (
    CAPITAL_POR_DEFECTO,
    ExperimentoNoTerminado,
    ServicioPaper,
    TransicionInvalida,
    receta_de_experimento,
)

router = APIRouter(prefix="/api/paper/interno", tags=["paper interno"])


class PeticionSesion(BaseModel):
    """Cuerpo de `POST /sesiones`: el origen (experimento o configuración) y los parámetros de la simulación forward."""

    id_experimento: str | None = None
    id_configuracion: str | None = None
    version: int | None = Field(None, ge=1, description="Versión de la configuración (la última si falta)")
    capital_inicial: float = Field(CAPITAL_POR_DEFECTO, gt=0)
    modo_ejecucion: Literal["cierre", "apertura_siguiente"] = "apertura_siguiente"
    comision: float | None = Field(None, ge=0, le=0.1)
    slippage: float | None = Field(None, ge=0, le=0.1)
    nombre: str | None = Field(None, max_length=120)
    fecha_alta: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")


def _servicio(ctx: Contexto) -> ServicioPaper:
    return ctx.servicio_paper_interno  # type: ignore[return-value]


def _sesion_o_404(servicio: ServicioPaper, id_sesion: str) -> None:
    if not servicio.existe(id_sesion):
        raise HTTPException(404, f"No existe la sesión {id_sesion}")


@router.get("/sesiones")
def listar_sesiones(ctx: Contexto = Depends(obtener_contexto)):
    return {"sesiones": _servicio(ctx).listar()}


@router.post("/sesiones", status_code=201)
def crear_sesion(peticion: PeticionSesion, ctx: Contexto = Depends(obtener_contexto)):
    if (peticion.id_experimento is None) == (peticion.id_configuracion is None):
        raise HTTPException(422, "Indica exactamente uno: id_experimento o id_configuracion")
    servicio = _servicio(ctx)
    try:
        if peticion.id_experimento is not None:
            try:
                receta = receta_de_experimento(ctx.almacen, peticion.id_experimento)
            except KeyError as error:
                raise HTTPException(404, f"No existe el experimento {peticion.id_experimento}") from error
        else:
            try:
                receta = ctx.almacen_configuraciones.leer(peticion.id_configuracion, peticion.version).receta()  # type: ignore[union-attr]
            except KeyError as error:
                raise HTTPException(404, f"No existe la configuración {peticion.id_configuracion}") from error
        sesion = servicio.crear_sesion(
            receta, nombre=peticion.nombre, fecha_alta=peticion.fecha_alta, capital_inicial=peticion.capital_inicial,
            modo_ejecucion=peticion.modo_ejecucion, comision=peticion.comision, slippage=peticion.slippage,
            id_experimento=peticion.id_experimento, id_configuracion=peticion.id_configuracion)
    except ExperimentoNoTerminado as error:
        raise HTTPException(409, str(error)) from error
    except KeyError as error:  # estrategia sin código registrado (p. ej. importada del TFM)
        raise HTTPException(422, f"La estrategia no tiene código registrado: {error}") from error
    except (ValueError, ValidationError) as error:
        raise HTTPException(422, str(error)) from error
    return servicio.detalle(sesion.id)


@router.get("/sesiones/{id_sesion}")
def detalle_sesion(id_sesion: str, ctx: Contexto = Depends(obtener_contexto)):
    servicio = _servicio(ctx)
    _sesion_o_404(servicio, id_sesion)
    return servicio.detalle(id_sesion)


@router.get("/sesiones/{id_sesion}/equity")
def equity_sesion(id_sesion: str, ctx: Contexto = Depends(obtener_contexto)):
    servicio = _servicio(ctx)
    _sesion_o_404(servicio, id_sesion)
    return {"equity": servicio.equity(id_sesion)}


@router.get("/sesiones/{id_sesion}/operaciones")
def operaciones_sesion(id_sesion: str, ctx: Contexto = Depends(obtener_contexto)):
    servicio = _servicio(ctx)
    _sesion_o_404(servicio, id_sesion)
    return {"operaciones": servicio.operaciones(id_sesion)}


@router.get("/sesiones/{id_sesion}/pesos")
def pesos_sesion(id_sesion: str, ctx: Contexto = Depends(obtener_contexto)):
    servicio = _servicio(ctx)
    _sesion_o_404(servicio, id_sesion)
    return {"pesos": servicio.pesos(id_sesion)}


@router.post("/sesiones/{id_sesion}/{accion}")
def controlar_sesion(id_sesion: str, accion: Literal["pausar", "reanudar", "detener"],
                     ctx: Contexto = Depends(obtener_contexto)):
    servicio = _servicio(ctx)
    _sesion_o_404(servicio, id_sesion)
    try:
        servicio.cambiar_estado(id_sesion, accion)
    except TransicionInvalida as error:
        raise HTTPException(409, str(error)) from error
    return servicio.detalle(id_sesion)


@router.post("/avanzar", status_code=202)
def avanzar(ctx: Contexto = Depends(obtener_contexto)):
    control = ctx.control_avance_paper
    if not control.iniciar():
        raise HTTPException(409, "Ya hay un avance en curso")
    ctx.ejecutor.submit(control.ejecutar, _servicio(ctx).avanzar)
    return control.estado()


@router.get("/avanzar/estado")
def estado_avance(ctx: Contexto = Depends(obtener_contexto)):
    return ctx.control_avance_paper.estado()


@router.get("/sesiones/{id_sesion}/espejo")
def espejo_sesion(id_sesion: str, ctx: Contexto = Depends(obtener_contexto)):
    """Curva real y backtest espejo (al cierre) de la misma receta y periodo; el espejo se calcula en segundo plano."""
    servicio = _servicio(ctx)
    _sesion_o_404(servicio, id_sesion)
    real = servicio.equity(id_sesion)
    config = servicio.config_espejo(id_sesion)
    if config is None:
        return {"estado": "sin_datos", "id_experimento": None, "real": real, "espejo": []}
    id_experimento = ctx.servicio.lanzar(config).id  # type: ignore[union-attr]
    estado = ctx.servicio.estado_de(id_experimento)  # type: ignore[union-attr]
    curva = ctx.almacen.leer_equity(id_experimento) if estado == "hecho" else None
    espejo = [] if curva is None else [
        {"fecha": f"{f:%Y-%m-%d}", "equity": float(e)} for f, e in zip(curva["fecha"], curva["equity"], strict=True)]
    return {"estado": estado, "id_experimento": id_experimento, "real": real, "espejo": espejo}

