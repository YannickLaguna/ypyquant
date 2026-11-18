"""
Rutas de estrategias y universos.

Responsabilidad:
- `GET /api/estrategias`: catálogo con el JSON Schema de los parámetros (la UI genera el formulario).
- `GET /api/estrategias/{nombre}`: detalle + código fuente en solo lectura.
- `POST /api/estrategias/recargar`: vuelve a descubrir `estrategias_usuario/` sin reiniciar la API.
- `GET /api/universos`: universos disponibles en `universos/`.

Se relaciona con: ypyquant.estrategias.registro y ypyquant.estrategias.experimentos (listar_universos).
"""

from fastapi import APIRouter, HTTPException

from ypyquant.estrategias import registro
from ypyquant.estrategias.experimentos import listar_universos

router = APIRouter(prefix="/api", tags=["estrategias"])


@router.get("/estrategias")
def listar_estrategias():
    return {"estrategias": registro.listar_estrategias(), "errores": registro.ERRORES_DESCUBRIMIENTO}


@router.get("/estrategias/{nombre}")
def detalle_estrategia(nombre: str):
    try:
        return {**registro.descripcion_estrategia(nombre), "codigo_fuente": registro.codigo_fuente(nombre)}
    except KeyError as error:
        raise HTTPException(404, str(error.args[0])) from error


@router.post("/estrategias/recargar")
def recargar_estrategias():
    nuevas = registro.descubrir_estrategias_usuario()
    return {"nuevas": nuevas, "errores": registro.ERRORES_DESCUBRIMIENTO}


@router.get("/universos")
def universos():
    return {"universos": listar_universos()}
