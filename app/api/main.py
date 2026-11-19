"""
Aplicación FastAPI de la interfaz interna de Sistema_trading.

Responsabilidad:
- `crear_app(...)`: fábrica que monta las rutas (estrategias, experimentos, barridos, paper trading), el
  contexto inyectable (almacén, cargador de precios, Alpaca, bot) y, si existe `app/web/dist`, sirve
  el frontend compilado en `/`. Es una capa fina: toda la lógica vive en `ypyquant.estrategias`.
- `app`: instancia lista para uvicorn.
- Seguridad local: sin autenticación, así que solo acepta `Host` 127.0.0.1/localhost (evita
  DNS-rebinding) y debe arrancarse escuchando en 127.0.0.1 (valor por defecto de uvicorn).

Arranque (desde la raíz del repo):
    uv run --extra ui uvicorn app.api.main:app --port 8010

Se relaciona con: app/api/contexto.py, rutas_*.py, paper.py y app/web (frontend, proxy /api -> 8010).
Variables: `YPYQUANT_PRECIOS=sintetico` usa precios sintéticos; `YPYQUANT_OFFLINE=1` no descarga;
`YPYQUANT_ESTRATEGIAS_USUARIO` cambia la carpeta de estrategias propias.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import rutas_barridos, rutas_estrategias, rutas_experimentos, rutas_paper
from app.api.contexto import Contexto
from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.data.matriz_precios import CargadorPrecios
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.barridos import AlmacenBarridos
from ypyquant.estrategias.precios import cargador_por_defecto

RAIZ_REPO = Path(__file__).resolve().parents[2]
CARPETA_WEB = RAIZ_REPO / "app" / "web" / "dist"


def crear_app(
    cargador_precios: CargadorPrecios | None = None,
    raiz_experimentos: Path | None = None,
    servicio_alpaca: ServicioAlpaca | None = None,
    bot: BotControl | None = None,
    cargar_env: bool = True,
    raiz_barridos: Path | None = None,
) -> FastAPI:
    """
    Construye la aplicación con sus dependencias (todas inyectables para los tests).

    Args:
        cargador_precios: por defecto `cargador_por_defecto()` (yfinance con caché Parquet).
        raiz_experimentos: carpeta del almacén (por defecto `data/experimentos/`).
        raiz_barridos: carpeta del almacén de barridos (por defecto `barridos/` junto a la de experimentos, es decir
            `data/barridos/`). El contador del periodo reservado vive en su carpeta padre (`reservado.json`).
        servicio_alpaca / bot: sustitutos para tests.
        cargar_env: si True, lee el `.env` de la raíz (claves de Alpaca) con python-dotenv.
    """
    if cargar_env:
        try:
            from dotenv import load_dotenv

            load_dotenv(RAIZ_REPO / ".env")
        except ImportError:
            pass

    api = FastAPI(title="Sistema_trading — interfaz interna", version="0.1.0")
    api.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    almacen = AlmacenExperimentos(raiz_experimentos)
    almacen.marcar_interrumpidos()  # lo que quedó en curso al caer el servidor anterior
    almacen_barridos = AlmacenBarridos(raiz_barridos or almacen.raiz.parent / "barridos", almacen)
    almacen_barridos.marcar_interrumpidos()
    api.state.ctx = Contexto(
        almacen=almacen,
        almacen_barridos=almacen_barridos,
        cargador_precios=cargador_precios or cargador_por_defecto(),
        alpaca=servicio_alpaca or ServicioAlpaca(),
        bot=bot or BotControl(),
    )

    @api.get("/api/salud", tags=["sistema"])
    def salud():
        return {"estado": "ok", "frontend": CARPETA_WEB.is_dir()}

    api.include_router(rutas_estrategias.router)
    api.include_router(rutas_experimentos.router)
    api.include_router(rutas_barridos.router)
    api.include_router(rutas_paper.router)

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
