"""
Rutas de universos: estado de las membresías del S&P 500 (universo point-in-time `sp500_pit`).

Responsabilidad (prefijo `/api/universos`):
- `GET /membresias/estado`: última comprobación contra las fuentes (Wikipedia como primaria, SPDR SPY como verificación),
  nº de miembros vigentes, cambios aplicados (total y los 10 últimos), cambios pendientes de revisar (las fuentes
  discrepan: no se aplican), antigüedad en días de la última comprobación correcta, avisos de calidad y si la
  comprobación automática diaria está activa (`YPYQUANT_MEMBRESIAS_AUTO`). Solo lee ficheros: no consulta la red.

Se relaciona con: ypyquant.data.membresias_sp500 (`ServicioMembresias.estado`), ypyquant.data.universos
(`avisos_membresias`), app/api/contexto.py (`servicio_membresias`) y app/web (tarjeta de membresías en «Paper interno»).
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends

from app.api.contexto import Contexto, obtener_contexto
from ypyquant.data.universos import avisos_membresias, cargar_membresias_sp500

router = APIRouter(prefix="/api/universos", tags=["universos"])


@router.get("/membresias/estado")
def estado_membresias(ctx: Contexto = Depends(obtener_contexto)):
    """Estado de las membresías del S&P 500 (ver cabecera del módulo)."""
    servicio = ctx.servicio_membresias
    estado = servicio.estado()
    try:
        membresias = cargar_membresias_sp500(servicio.ruta)
        estado["avisos_calidad"] = avisos_membresias(membresias, datetime.now().date().isoformat(),
                                                     comprobado=estado["ultima_comprobacion_correcta"])
    except FileNotFoundError as error:
        estado["avisos_calidad"] = [str(error)]
    return estado
