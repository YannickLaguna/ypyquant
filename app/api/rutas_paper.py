"""
Rutas de paper trading (Alpaca, cuenta paper) y control del bot.

Responsabilidad:
- Lectura: `/api/paper/estado`, `/cuenta`, `/posiciones`, `/ordenes`, `/historial`, `/fills`.
  Sin claves de Alpaca responden 503 con un mensaje claro (la UI lo muestra); los errores de
  Alpaca se traducen a 502.
- Bot: `POST /api/paper/bot/arrancar` (estrategia del registro + parámetros + tickers),
  `POST /api/paper/bot/parar`, `GET /api/paper/bot/estado` y `GET /api/paper/bot/log?desde=N` (polling por línea).

Se relaciona con: app/api/paper.py (ServicioAlpaca, BotControl), app/api/contexto.py y el registro
de estrategias (se valida el nombre y los parámetros antes de lanzar el subproceso).
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.api.contexto import Contexto, obtener_contexto
from app.api.paper import SinClavesAlpaca, leer_fills
from ypyquant.estrategias import registro

router = APIRouter(prefix="/api/paper", tags=["paper trading"])


class PeticionBot(BaseModel):
    """Qué estrategia ejecutar en paper: nombre del registro, parámetros, universo e intervalo entre iteraciones."""

    estrategia: str
    parametros: dict[str, Any] = Field(default_factory=dict)
    tickers: list[str] = Field(min_length=1)
    intervalo_s: int = Field(60, ge=5, le=86_400)


def _alpaca(ctx: Contexto, accion):
    """Ejecuta `accion(servicio)` traduciendo 'sin claves' a 503 y los fallos de Alpaca a 502."""
    try:
        return accion(ctx.alpaca)
    except SinClavesAlpaca as error:
        raise HTTPException(503, str(error)) from error
    except Exception as error:
        raise HTTPException(502, f"Error de Alpaca: {error}") from error


@router.get("/estado")
def estado(ctx: Contexto = Depends(obtener_contexto)):
    return {"alpaca_configurado": ctx.alpaca.disponible, "bot": ctx.bot.estado()}


@router.get("/cuenta")
def cuenta(ctx: Contexto = Depends(obtener_contexto)):
    return _alpaca(ctx, lambda s: s.cuenta())


@router.get("/posiciones")
def posiciones(ctx: Contexto = Depends(obtener_contexto)):
    return {"posiciones": _alpaca(ctx, lambda s: s.posiciones())}


@router.get("/ordenes")
def ordenes(limite: int = 100, ctx: Contexto = Depends(obtener_contexto)):
    return {"ordenes": _alpaca(ctx, lambda s: s.ordenes(limite))}


@router.get("/historial")
def historial(periodo: str = "1M", marco: str = "1D", ctx: Contexto = Depends(obtener_contexto)):
    return _alpaca(ctx, lambda s: s.historial_equity(periodo, marco))


@router.get("/fills")
def fills(limite: int = 200):
    return {"fills": leer_fills(limite=limite)}


@router.post("/bot/arrancar")
def arrancar_bot(peticion: PeticionBot, ctx: Contexto = Depends(obtener_contexto)):
    if not ctx.alpaca.disponible:
        raise HTTPException(503, "Faltan ALPACA_API_KEY y ALPACA_SECRET_KEY: el bot no puede operar en paper")
    try:
        clase = registro.obtener_estrategia(peticion.estrategia)
        parametros = clase.Parametros(**peticion.parametros).model_dump()
    except KeyError as error:
        raise HTTPException(404, str(error.args[0])) from error
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error
    try:
        return ctx.bot.arrancar(peticion.estrategia, parametros, [t.upper() for t in peticion.tickers], peticion.intervalo_s)
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error


@router.post("/bot/parar")
def parar_bot(ctx: Contexto = Depends(obtener_contexto)):
    return ctx.bot.parar()


@router.get("/bot/estado")
def estado_bot(ctx: Contexto = Depends(obtener_contexto)):
    return ctx.bot.estado()


@router.get("/bot/log")
def log_bot(desde: int = 0, ctx: Contexto = Depends(obtener_contexto)):
    return ctx.bot.log(max(desde, 0))
