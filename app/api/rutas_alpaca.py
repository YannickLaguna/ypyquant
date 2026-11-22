"""
Rutas de Alpaca multi-perfil: qué cuentas hay, y un bot por cuenta vinculado a un experimento.

Responsabilidad (prefijo `/api/paper/alpaca`; los datos de cada cuenta siguen en `/api/paper/cuenta?perfil=X`, etc.):
- `GET /perfiles`: las cuentas detectadas en el entorno (`ALPACA_<PERFIL>_API_KEY` / `_SECRET_KEY`; `ALPACA_API_KEY` es
  `DEFECTO`) más las declaradas en `ALPACA_PERFILES`. Por cada una: nombre, si está configurada, los 4 últimos
  caracteres de la API key (NUNCA el secreto), las variables que faltan, el estado de conexión con Alpaca y equity, el
  bot (en marcha o no) y el vínculo (experimento al que opera). Incluye el límite blando de cuentas activas.
- `GET /bots/{perfil}` (estado + vínculo), `POST /bots/{perfil}/arrancar` (experimento o configuración, modo
  `continuo`|`diario` e intervalo), `POST /bots/{perfil}/parar`, `GET /bots/{perfil}/log?desde=N` (log redactado) y
  `GET /{perfil}/instantaneas` (equity y posiciones diarias de la cuenta).
- Errores: 503 si el perfil no tiene claves, 404 si el experimento no existe, 422 si su receta no se puede operar
  (estrategia o parámetros inválidos, universo dinámico), 409 si el perfil está ocupado o se supera el límite blando.

Se relaciona con: app/api/gestor_bots.py (los procesos), app/api/paper.py (ServicioAlpaca), app/api/contexto.py
(`alpaca_de`), ypyquant.perfiles_alpaca (perfiles, vínculo, almacén) y app/web (página Paper trading).
"""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.api.contexto import Contexto, obtener_contexto
from app.api.gestor_bots import LimiteBotsActivos, PerfilOcupado
from app.api.rutas_paper import servicio_de_perfil
from ypyquant.perfiles_alpaca.perfiles import LIMITE_PERFILES_ACTIVOS, NOMBRE_DEFECTO, descubrir_perfiles
from ypyquant.perfiles_alpaca.vinculo import RecetaNoOperable, receta_de_origen, validar_operable

router = APIRouter(prefix="/api/paper/alpaca", tags=["paper trading: Alpaca multi-perfil"])


class PeticionBotPerfil(BaseModel):
    """Qué opera el bot de un perfil: un experimento (o una configuración guardada), modo e intervalo entre iteraciones."""

    experimento_id: str | None = None
    configuracion_id: str | None = None
    modo: Literal["continuo", "diario"] = "continuo"
    intervalo_s: int = Field(60, ge=5, le=86_400)

    @model_validator(mode="after")
    def _un_origen(self):
        if bool(self.experimento_id) == bool(self.configuracion_id):
            raise ValueError("indica exactamente uno de experimento_id o configuracion_id")
        return self


def _resumen_vinculo(vinculo: dict[str, Any] | None) -> dict[str, Any] | None:
    """Lo que la UI necesita del vínculo (sin el JSON completo de la receta)."""
    if not vinculo:
        return None
    return {k: vinculo.get(k) for k in ("origen", "estrategia", "huella_receta", "frecuencia_rebalanceo", "tickers",
                                         "fecha_alta", "equity_inicial")}


def _perfiles_visibles(ctx: Contexto) -> list[dict[str, Any]]:
    """Perfiles del entorno más los servicios inyectados (tests) que estén disponibles, DEFECTO primero."""
    base = {p.nombre: p.a_dict() for p in descubrir_perfiles()}
    for nombre, servicio in ctx.alpacas.items():
        if nombre not in base and servicio.disponible:
            base[nombre] = {"nombre": nombre, "configurado": True, "ultimos4": None, "faltan": [], "variables": []}
    for nombre, datos in base.items():
        if not datos["configurado"] and servicio_de_perfil(ctx, nombre).disponible:  # claves inyectadas (tests)
            datos.update(configurado=True, faltan=[])
    return sorted(base.values(), key=lambda p: (p["nombre"] != NOMBRE_DEFECTO, p["nombre"]))


@router.get("/perfiles")
def perfiles(conexion: bool = True, ctx: Contexto = Depends(obtener_contexto)):
    """Cuentas de Alpaca con su conexión, bot y vínculo. `conexion=false` evita consultar a Alpaca."""
    filas = []
    for datos in _perfiles_visibles(ctx):
        nombre = datos["nombre"]
        servicio = servicio_de_perfil(ctx, nombre)
        estado_conexion = (servicio.comprobar_conexion() if conexion
                           else {"configurada": datos["configurado"], "conectada": None, "equity": None, "error": None})
        filas.append({**datos, "conexion": estado_conexion, "bot": ctx.gestor_bots.estado(nombre),
                      "vinculo": _resumen_vinculo(ctx.almacen_perfiles.leer_vinculo(nombre))})
    activos = ctx.gestor_bots.activos()
    configurados = sum(1 for f in filas if f["configurado"])
    return {"perfiles": filas, "limite_activos": LIMITE_PERFILES_ACTIVOS, "activos": activos,
            "aviso": (f"Hay {configurados} perfiles configurados; el límite blando son {LIMITE_PERFILES_ACTIVOS} bots a la vez."
                      if configurados > LIMITE_PERFILES_ACTIVOS else None)}


@router.get("/bots/{perfil}")
def estado_bot_perfil(perfil: str, ctx: Contexto = Depends(obtener_contexto)):
    servicio_de_perfil(ctx, perfil)
    return {**ctx.gestor_bots.estado(perfil), "vinculo": _resumen_vinculo(ctx.almacen_perfiles.leer_vinculo(perfil))}


@router.post("/bots/{perfil}/arrancar")
def arrancar_bot_perfil(perfil: str, peticion: PeticionBotPerfil, ctx: Contexto = Depends(obtener_contexto)):
    servicio = servicio_de_perfil(ctx, perfil)
    if not servicio.disponible:
        raise HTTPException(503, f"Perfil {perfil} sin configurar: el bot no puede operar (ver /api/paper/alpaca/perfiles)")
    tipo, id_origen = (("experimento", peticion.experimento_id) if peticion.experimento_id
                       else ("configuracion", peticion.configuracion_id))
    try:
        receta = receta_de_origen(tipo, id_origen, ctx.almacen, ctx.almacen_configuraciones)
        validar_operable(receta)  # falla pronto (422): universo dinámico o apalancamiento > 1
    except KeyError as error:
        raise HTTPException(404, str(error.args[0])) from error
    except RecetaNoOperable as error:
        raise HTTPException(422, str(error)) from error
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False, include_context=False)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if perfil == NOMBRE_DEFECTO and ctx.bot.en_marcha():
        raise HTTPException(409, "El bot heredado (/api/paper/bot) está en marcha sobre la misma cuenta; páralo antes")
    try:
        return ctx.gestor_bots.arrancar(perfil, peticion.experimento_id, peticion.configuracion_id, peticion.modo,
                                        peticion.intervalo_s)
    except (PerfilOcupado, LimiteBotsActivos) as error:
        raise HTTPException(409, str(error)) from error


@router.post("/bots/{perfil}/parar")
def parar_bot_perfil(perfil: str, ctx: Contexto = Depends(obtener_contexto)):
    servicio_de_perfil(ctx, perfil)
    return ctx.gestor_bots.parar(perfil)


@router.get("/bots/{perfil}/log")
def log_bot_perfil(perfil: str, desde: int = 0, ctx: Contexto = Depends(obtener_contexto)):
    servicio_de_perfil(ctx, perfil)
    return ctx.gestor_bots.log(perfil, max(desde, 0))


@router.get("/{perfil}/instantaneas")
def instantaneas(perfil: str, ctx: Contexto = Depends(obtener_contexto)):
    """Instantáneas diarias de la cuenta (fecha, equity, efectivo y posiciones) que guarda el bot; más antiguas primero."""
    servicio_de_perfil(ctx, perfil)
    tabla = ctx.almacen_perfiles.leer_instantaneas(perfil)
    return {"instantaneas": tabla.to_dict(orient="records")}
