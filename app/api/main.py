"""
Aplicación FastAPI de la interfaz interna de Sistema_trading.

Responsabilidad:
- `crear_app(...)`: fábrica que monta las rutas (estrategias, experimentos, barridos, configuraciones, marcos, paper trading y paper interno), el
  contexto inyectable (almacén, cargador de precios, Alpaca, bot) y, si existe `app/web/dist`, sirve
  el frontend compilado en `/`. Es una capa fina: toda la lógica vive en `ypyquant.estrategias`.
- `app`: instancia lista para uvicorn.
- Seguridad local: sin autenticación, así que solo acepta `Host` 127.0.0.1/localhost (evita
  DNS-rebinding) y debe arrancarse escuchando en 127.0.0.1 (valor por defecto de uvicorn).

Arranque (desde la raíz del repo):
    uv run --extra ui uvicorn app.api.main:app --port 8010

Se relaciona con: app/api/contexto.py, rutas_*.py, paper.py y app/web (frontend, proxy /api -> 8010).
Variables: `YPYQUANT_PRECIOS=sintetico` usa precios sintéticos; `YPYQUANT_OFFLINE=1` no descarga;
`YPYQUANT_ESTRATEGIAS_USUARIO` cambia la carpeta de estrategias propias; `YPYQUANT_PAPER_INTERVALO_MIN` (30 por
defecto, 0 lo desactiva) fija cada cuántos minutos el temporizador del paper interno comprueba si hay días cerrados
pendientes y avanza las sesiones activas (ver ypyquant/paper/planificador.py; vive en el lifespan de la API).
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import (
    rutas_barridos,
    rutas_configuraciones,
    rutas_estrategias,
    rutas_experimentos,
    rutas_marcos,
    rutas_paper,
    rutas_paper_interno,
    rutas_universos,
)
from app.api import rutas_alpaca
from app.api.contexto import Contexto
from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.data.membresias_sp500 import ServicioMembresias, comprobacion_automatica_activa
from ypyquant.paper.planificador import PlanificadorPaper
from ypyquant.data.matriz_precios import CargadorPrecios
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.barridos import AlmacenBarridos
from ypyquant.estrategias.configuraciones import AlmacenConfiguraciones
from ypyquant.estrategias.marcos import AlmacenMarcos
from ypyquant.raiz_datos import RaizDatos
from ypyquant.data.fuentes import fuente_por_defecto

logger = logging.getLogger(__name__)

RAIZ_REPO = Path(__file__).resolve().parents[2]
INTERVALO_PAPER_MIN_POR_DEFECTO = 30.0
CARPETA_WEB = RAIZ_REPO / "app" / "web" / "dist"


def _intervalo_paper_s(intervalo_min: float | None) -> float:
    """Intervalo del temporizador del paper interno en segundos: el argumento, o `YPYQUANT_PAPER_INTERVALO_MIN`, o 30 min."""
    if intervalo_min is None:
        crudo = os.getenv("YPYQUANT_PAPER_INTERVALO_MIN")
        try:
            intervalo_min = float(crudo) if crudo else INTERVALO_PAPER_MIN_POR_DEFECTO
        except ValueError:
            logger.warning("YPYQUANT_PAPER_INTERVALO_MIN=%r no es un número: se usan %s minutos", crudo, INTERVALO_PAPER_MIN_POR_DEFECTO)
            intervalo_min = INTERVALO_PAPER_MIN_POR_DEFECTO
    return max(intervalo_min, 0.0) * 60


def crear_app(
    cargador_precios: CargadorPrecios | None = None,
    raiz_experimentos: Path | None = None,
    servicio_alpaca: ServicioAlpaca | None = None,
    bot: BotControl | None = None,
    cargar_env: bool = True,
    raiz_barridos: Path | None = None,
    raiz_configuraciones: Path | None = None,
    raiz_marcos: Path | None = None,
    catchup_paper_al_arrancar: bool = True,
    reloj_paper=None,
    intervalo_paper_min: float | None = None,
    servicio_membresias: ServicioMembresias | None = None,
) -> FastAPI:
    """
    Construye la aplicación con sus dependencias (todas inyectables para los tests).

    Args:
        cargador_precios: por defecto `fuente_por_defecto()` (yfinance con caché Parquet).
        raiz_experimentos: carpeta del almacén (por defecto `data/experimentos/`).
        raiz_barridos: carpeta del almacén de barridos (por defecto `barridos/` junto a la de experimentos, es decir
            `data/barridos/`). El contador del periodo reservado vive en su carpeta padre (`reservado.json`).
        raiz_configuraciones / raiz_marcos: carpetas de las configuraciones guardadas y de los marcos de comparación
            (por defecto `configuraciones/` y `marcos/` junto a la del almacén, es decir `data/configuraciones/` y `data/marcos/`).
        servicio_alpaca / bot: sustitutos para tests.
        cargar_env: si True, lee el `.env` de la raíz (claves de Alpaca) con python-dotenv.
        catchup_paper_al_arrancar: si hay sesiones de paper interno activas, las pone al día en segundo plano al
            arrancar (los tests lo desactivan). reloj_paper: reloj inyectable del paper interno (por defecto, Nueva York).
        intervalo_paper_min: cada cuántos minutos el temporizador del paper interno avanza las sesiones con días cerrados
            pendientes mientras la API vive; 0 lo desactiva y None lo lee de `YPYQUANT_PAPER_INTERVALO_MIN` (30 por defecto).
            Solo arranca dentro del ciclo de vida de la app (lifespan), así que un cliente de pruebas sin `with` no lo lanza.
    """
    if cargar_env:
        try:
            from dotenv import load_dotenv

            load_dotenv(RAIZ_REPO / ".env")
        except ImportError:
            pass

    @asynccontextmanager
    async def ciclo_de_vida(_: FastAPI) -> AsyncIterator[None]:
        """Arranca el temporizador del paper interno con la API y lo detiene limpio al cerrarla."""
        planificador = api.state.ctx.planificador_paper
        if planificador is not None and planificador.intervalo_s > 0:
            planificador.iniciar()
        try:
            yield
        finally:
            if planificador is not None:
                planificador.detener()

    api = FastAPI(title="Sistema_trading — interfaz interna", version="0.1.0", lifespan=ciclo_de_vida)
    api.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    raiz = RaizDatos.de_experimentos(raiz_experimentos) if raiz_experimentos else RaizDatos()
    almacen = AlmacenExperimentos(raiz.experimentos)
    almacen.marcar_interrumpidos()  # lo que quedó en curso al caer el servidor anterior
    # con una carpeta de barridos propia el contador del reservado sigue junto a ella (su carpeta padre)
    almacen_barridos = AlmacenBarridos(raiz_barridos or raiz.barridos, almacen, None if raiz_barridos else raiz.reservado)
    almacen_barridos.marcar_interrumpidos()
    almacen_marcos = AlmacenMarcos(raiz_marcos or raiz.marcos)
    api.state.ctx = Contexto(
        almacen=almacen,
        almacen_barridos=almacen_barridos,
        almacen_configuraciones=AlmacenConfiguraciones(raiz_configuraciones or raiz.configuraciones),
        almacen_marcos=almacen_marcos,
        cargador_precios=cargador_precios or fuente_por_defecto(),
        alpaca=servicio_alpaca or ServicioAlpaca(),
        bot=bot or BotControl(),
        **({"reloj_paper": reloj_paper} if reloj_paper else {}),
    )

    # membresías del S&P 500: el planificador las refresca una vez al día; YPYQUANT_MEMBRESIAS_AUTO=0 lo desactiva
    api.state.ctx.servicio_membresias = servicio_membresias or ServicioMembresias(
        activo=comprobacion_automatica_activa(os.environ))
    api.state.ctx.planificador_paper = PlanificadorPaper(
        api.state.ctx.servicio_paper_interno, api.state.ctx.control_avance_paper,
        intervalo_s=_intervalo_paper_s(intervalo_paper_min), membresias=api.state.ctx.servicio_membresias)

    api.state.ctx.gestor_bots.rehidratar()  # bots de Alpaca que siguieron vivos tras reiniciar la API
    almacen.indexar()  # índice receta -> experimentos (backtests de una configuración sin recorrer el disco)
    almacen_marcos.indexar_comparativas()  # índice configuración -> comparativas

    @api.get("/api/salud", tags=["sistema"])
    def salud():
        return {"estado": "ok", "frontend": CARPETA_WEB.is_dir()}

    api.include_router(rutas_estrategias.router)
    api.include_router(rutas_experimentos.router)
    api.include_router(rutas_barridos.router)
    api.include_router(rutas_configuraciones.router)
    api.include_router(rutas_marcos.router)
    api.include_router(rutas_paper.router)
    api.include_router(rutas_alpaca.router)
    api.include_router(rutas_paper_interno.router)
    api.include_router(rutas_universos.router)

    # Paper interno: catch-up de los días perdidos (PC apagado) sin bloquear el arranque
    servicio_paper = api.state.ctx.servicio_paper_interno
    if catchup_paper_al_arrancar and servicio_paper.hay_sesiones_activas() and api.state.ctx.control_avance_paper.iniciar():
        api.state.ctx.ejecutor.submit(api.state.ctx.control_avance_paper.ejecutar, servicio_paper.avanzar)

    if CARPETA_WEB.is_dir():  # frontend compilado (npm run build); en desarrollo se usa `npm run dev`
        api.mount("/assets", StaticFiles(directory=CARPETA_WEB / "assets"), name="assets")

        @api.get("/{ruta:path}", include_in_schema=False)
        def frontend(ruta: str):
            if ruta.startswith("api/"):
                raise HTTPException(404, "Ruta de API inexistente")
            archivo = (CARPETA_WEB / ruta).resolve()
            if ruta and archivo.is_file() and CARPETA_WEB.resolve() in archivo.parents:
                return FileResponse(archivo)
            return FileResponse(CARPETA_WEB / "index.html")  # SPA: el router del cliente decide

    return api


_app_global: FastAPI | None = None


def __getattr__(nombre: str):
    """`app.api.main:app` para uvicorn, creada al pedirla (importar el módulo no toca disco ni lee el .env)."""
    global _app_global
    if nombre == "app":
        if _app_global is None:
            _app_global = crear_app()
        return _app_global
    raise AttributeError(nombre)
