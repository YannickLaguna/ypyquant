"""
Registro y descubrimiento automático de estrategias.

Responsabilidad:
- `@registrar_estrategia`: decorador de clase que añade una `Estrategia` al registro global.
- `listar_estrategias()`, `obtener_estrategia(nombre)`, `descripcion_estrategia(nombre)` y
  `codigo_fuente(nombre)` (solo lectura, para mostrarlo en la UI).
- `descubrir_estrategias_usuario(carpeta)`: importa todos los `.py` de una carpeta
  (por defecto `estrategias_usuario/` en la raíz del repo, o la de la variable de entorno
  `YPYQUANT_ESTRATEGIAS_USUARIO`) para que crear una estrategia sea escribir un archivo.
  Los fallos de importación no tumban nada: quedan en `ERRORES_DESCUBRIMIENTO`.
- `cargar_estrategias()`: carga las incluidas (`incluidas.py`) y descubre las de usuario; es
  idempotente y es lo que llaman la API y `ejecutar_experimento`.

Se relaciona con:
- ypyquant.estrategias.base (Estrategia), .incluidas (las registra al importarse).
- ypyquant.estrategias.experimentos (obtener_estrategia) y app/api (listado y detalle).
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any

from ypyquant.estrategias.base import Estrategia

logger = logging.getLogger(__name__)

# Raíz del repo (src/ypyquant/estrategias/registro.py -> parents[3]); carpeta de estrategias propias
RAIZ_REPO = Path(__file__).resolve().parents[3]
CARPETA_USUARIO_POR_DEFECTO = RAIZ_REPO / "estrategias_usuario"
VARIABLE_ENTORNO_CARPETA = "YPYQUANT_ESTRATEGIAS_USUARIO"

_REGISTRO: dict[str, type[Estrategia]] = {}
ERRORES_DESCUBRIMIENTO: dict[str, str] = {}
_cargadas: set[str] = set()


def registrar_estrategia(clase: type[Estrategia]) -> type[Estrategia]:
    """Decorador: registra `clase` bajo `clase.nombre`. Falla si falta el nombre o ya lo usa otra clase."""
    nombre = getattr(clase, "nombre", None)
    if not nombre or not isinstance(nombre, str):
        raise ValueError(f"{clase.__name__} debe definir el atributo de clase `nombre`")
    existente = _REGISTRO.get(nombre)
    if existente is not None and (existente.__module__, existente.__qualname__) != (clase.__module__, clase.__qualname__):
        raise ValueError(f"Ya existe una estrategia registrada con el nombre '{nombre}' ({existente.__module__})")
    _REGISTRO[nombre] = clase
    return clase


def obtener_estrategia(nombre: str) -> type[Estrategia]:
    """Clase registrada con ese nombre; `KeyError` con las disponibles si no existe."""
    cargar_estrategias()
    if nombre not in _REGISTRO:
        raise KeyError(f"Estrategia '{nombre}' no registrada. Disponibles: {', '.join(sorted(_REGISTRO))}")
    return _REGISTRO[nombre]


def _resumen(clase: type[Estrategia]) -> dict[str, Any]:
    """Metadatos serializables de una estrategia (incluye el JSON Schema de sus parámetros)."""
    return {
        "nombre": clase.nombre,
        "descripcion": clase.descripcion,
        "origen": clase.origen,
        "archivo": clase.archivo,
        "frecuencia_rebalanceo": clase.frecuencia_rebalanceo,
        "universo_sugerido": clase.universo_sugerido,
        "parametros_schema": clase.Parametros.model_json_schema(),
        "parametros_defecto": clase.Parametros().model_dump(),
    }


def listar_estrategias() -> list[dict[str, Any]]:
    """Todas las estrategias registradas (incluidas + usuario), ordenadas por nombre."""
    cargar_estrategias()
    return [_resumen(_REGISTRO[nombre]) for nombre in sorted(_REGISTRO)]


def descripcion_estrategia(nombre: str) -> dict[str, Any]:
    """Metadatos de una estrategia concreta."""
    return _resumen(obtener_estrategia(nombre))


def codigo_fuente(nombre: str) -> str:
    """Código fuente completo del módulo donde se define la estrategia (solo lectura)."""
    clase = obtener_estrategia(nombre)
    try:
        return inspect.getsource(sys.modules[clase.__module__])
    except (OSError, TypeError, KeyError):
        try:
            return inspect.getsource(clase)
        except (OSError, TypeError):
            return "# Código fuente no disponible"


def carpeta_usuario() -> Path:
    """Carpeta de estrategias de usuario (variable de entorno o `estrategias_usuario/` del repo)."""
    return Path(os.environ.get(VARIABLE_ENTORNO_CARPETA) or CARPETA_USUARIO_POR_DEFECTO)


def descubrir_estrategias_usuario(carpeta: Path | str | None = None) -> list[str]:
    """
    Importa cada `.py` de la carpeta (ignora los que empiezan por `_`) y devuelve los nombres nuevos.

    Las estrategias registradas desde esos archivos quedan con `origen='usuario'` y `archivo`.
    Un archivo con errores se anota en `ERRORES_DESCUBRIMIENTO[archivo]` y se sigue con el resto.
    """
    carpeta = Path(carpeta) if carpeta else carpeta_usuario()
    nuevos: list[str] = []
    if not carpeta.is_dir():
        return nuevos
    for ruta in sorted(carpeta.glob("*.py")):
        if ruta.name.startswith("_"):
            continue
        antes = set(_REGISTRO)
        nombre_modulo = f"estrategia_usuario_{ruta.stem}"
        try:
            especificacion = importlib.util.spec_from_file_location(nombre_modulo, ruta)
            modulo = importlib.util.module_from_spec(especificacion)
            sys.modules[nombre_modulo] = modulo  # inspect.getsource necesita el módulo en sys.modules
            especificacion.loader.exec_module(modulo)
            ERRORES_DESCUBRIMIENTO.pop(str(ruta), None)
        except Exception as error:  # noqa: BLE001 - un archivo de usuario roto no debe tumbar el registro
            sys.modules.pop(nombre_modulo, None)
            ERRORES_DESCUBRIMIENTO[str(ruta)] = f"{type(error).__name__}: {error}"
            logger.warning("No se pudo cargar la estrategia de usuario %s: %s", ruta.name, error)
            continue
        for nombre in set(_REGISTRO) - antes:
            _REGISTRO[nombre].origen = "usuario"
            _REGISTRO[nombre].archivo = str(ruta)
            nuevos.append(nombre)
    return sorted(nuevos)


def cargar_estrategias(recargar_usuario: bool = False) -> None:
    """Carga las estrategias incluidas y descubre las de usuario (una sola vez salvo `recargar_usuario`)."""
    if "incluidas" not in _cargadas:
        _cargadas.add("incluidas")
        from ypyquant.estrategias import incluidas  # noqa: F401 - registran las estrategias de serie
    clave = f"usuario:{carpeta_usuario()}"
    if recargar_usuario or clave not in _cargadas:
        _cargadas.add(clave)
        descubrir_estrategias_usuario()
