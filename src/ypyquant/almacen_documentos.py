"""
Almacén de documentos por id: el módulo común de persistencia en disco de la librería.

Responsabilidad: guardar «documentos» (una carpeta por id con varios archivos JSON y Parquet dentro) con las garantías
que antes reimplementaba cada almacén por su cuenta:
- escritura atómica (archivo temporal único por proceso e hilo + `os.replace`) con reintentos ante el `PermissionError`
  transitorio de Windows (antivirus, Syncthing, otro hilo leyendo en ese instante);
- lectura que tolera esos mismos bloqueos (y, opcionalmente, un archivo que aún no «se ve» tras el reemplazo);
- validación del id (nunca sale de la raíz), existencia, listado de ids, borrado, y la limpieza NumPy/NaN -> JSON válido.

Interfaz (tres piezas, todas sin conocimiento del dominio):
- `AlmacenDocumentos(raiz, nombre=..., validar_id=..., marca=...)`: `<raiz>/<id>/<archivo>.json|.parquet`. `crear`,
  `existe`, `ids`, `borrar`, `escribir_json`/`leer_json`/`tiene_json`/`nombres_json`, `escribir_parquet`/`leer_parquet`/
  `tiene_parquet`/`ruta_parquet`, y un `cerrojo` reentrante para las secuencias leer-modificar-escribir del adaptador.
  `marca` es el JSON que define que una carpeta ES un documento (p. ej. `barrido`): sin ella, toda carpeta cuenta.
- `ColeccionJson(raiz, prefijo=...)`: lo mismo para archivos planos `<raiz>/<id>.json` (sin carpeta por documento).
- `escribir_json_atomico(ruta, contenido)` / `leer_json(ruta)`: las primitivas, para un archivo suelto (`reservado.json`).
Los validadores de id listos para usar son `id_sin_separadores` (ids generados por huella) e `id_slug` (ids elegidos por
el usuario); un validador es una función `str -> bool`.

Servirá igual para el paper trading interno (sesiones, órdenes, fills): un adaptador fino con su `nombre`, su validador y
su `marca`, sin tocar este módulo. El formato en disco (nombres de archivo, JSON indentado con 2 espacios y sin escapar
tildes, Parquet sin índice por defecto) es el histórico: `tests/test_formato_en_disco.py` lo fija.

Se relaciona con:
- ypyquant.raiz_datos (de dónde cuelga cada almacén dentro de `data/`).
- ypyquant.estrategias.almacen / barridos / configuraciones / marcos / reservado (los adaptadores de dominio) y
  .importacion_tfm (escribe experimentos con el mismo formato).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REINTENTOS_ARCHIVO = 20  # ~2 s en total ante bloqueos transitorios de Windows

_ID_SLUG = re.compile(r"[a-z0-9][a-z0-9\-_.]*")


# ---- ids -----------------------------------------------------------------------------------------


def id_sin_separadores(identificador: str) -> bool:
    """Id válido si no está vacío, no tiene separadores de ruta y no empieza por punto (ids generados por huella)."""
    return bool(identificador) and "/" not in identificador and "\\" not in identificador and not identificador.startswith(".")


def id_slug(identificador: str) -> bool:
    """Id válido si es un slug: minúsculas, dígitos, `-`, `_` y `.`, empezando por letra o dígito y sin `..`."""
    return bool(identificador) and _ID_SLUG.fullmatch(identificador) is not None and ".." not in identificador


# ---- primitivas ----------------------------------------------------------------------------------


def limpiar_para_json(valor: Any) -> Any:
    """Convierte tipos NumPy a nativos y NaN/inf a None (JSON válido), recursivamente."""
    if isinstance(valor, dict):
        return {k: limpiar_para_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [limpiar_para_json(v) for v in valor]
    if isinstance(valor, (np.floating, float)):
        return None if not math.isfinite(float(valor)) else float(valor)
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.bool_):
        return bool(valor)
    return valor


def reintentar_en_windows(operacion: Callable[[], Any]) -> Any:
    """
    Ejecuta `operacion` reintentando ante `PermissionError`.

    En Windows, `os.replace` falla si otro hilo tiene el destino abierto en ese instante, y una lectura falla si coincide
    con el reemplazo (fallo intermitente de `tests/test_api.py` del 2026-11-19). Son bloqueos de milisegundos: se
    reintenta con espera creciente.
    """
    for intento in range(REINTENTOS_ARCHIVO):
        try:
            return operacion()
        except PermissionError:
            if intento == REINTENTOS_ARCHIVO - 1:
                raise
            time.sleep(0.01 * (intento + 1))
    return None


def _temporal_de(ruta: Path) -> Path:
    """Temporal único por proceso e hilo: dos hilos escribiendo el mismo archivo no comparten el `.tmp`."""
    return ruta.with_name(f"{ruta.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def escribir_json_atomico(ruta: Path, contenido: Any) -> None:
    """Escritura atómica de un JSON (temporal + reemplazo): otro hilo puede leer mientras se escribe."""
    temporal = _temporal_de(ruta)
    temporal.write_text(json.dumps(limpiar_para_json(contenido), ensure_ascii=False, indent=2), encoding="utf-8")
    reintentar_en_windows(lambda: os.replace(temporal, ruta))


def escribir_parquet_atomico(tabla: pd.DataFrame, ruta: Path, index: bool = False) -> None:
    """
    Escritura atómica de un Parquet (temporal + reemplazo). Sin ella, una lectura concurrente podía abrir el archivo a
    medio escribir: pyarrow lanza `ArrowInvalid` (un `ValueError`) y la API respondía 404 (fallo intermitente de
    `tests/test_api.py`, 2026-11-19).
    """
    temporal = _temporal_de(ruta)
    tabla.to_parquet(temporal, index=index)
    reintentar_en_windows(lambda: os.replace(temporal, ruta))


def leer_json(ruta: Path, reintentar_si_falta: bool = False) -> Any:
    """
    Lee un JSON (None si no existe).

    `reintentar_si_falta`: para archivos que deben existir (p. ej. `estado.json` de un experimento con `config.json`): en
    Windows el destino de `os.replace` puede no verse durante un instante.
    """
    for intento in range(REINTENTOS_ARCHIVO):
        try:
            return json.loads(ruta.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if not reintentar_si_falta or intento == REINTENTOS_ARCHIVO - 1:
                return None
        except PermissionError:
            if intento == REINTENTOS_ARCHIVO - 1:
                raise
        time.sleep(0.01 * (intento + 1))
    return None


# ---- carpeta por documento -----------------------------------------------------------------------


class AlmacenDocumentos:
    """
    Documentos en `<raiz>/<id>/`, cada uno con sus archivos `<nombre>.json` y `<nombre>.parquet`.

    Args:
        raiz: carpeta del almacén (se crea si no existe).
        nombre: cómo se llama el documento en los mensajes de error («experimento», «barrido»...).
        validar_id: `str -> bool` (por defecto `id_sin_separadores`); un id inválido lanza `ValueError` en toda operación.
        marca: nombre (sin extensión) del JSON que define que una carpeta es un documento. Con ella, `existe` e `ids`
            solo cuentan las carpetas que la tienen; sin ella, cualquier carpeta cuenta.

    `cerrojo` (reentrante) es de uso voluntario: el adaptador lo toma en las secuencias leer-modificar-escribir.
    """

    def __init__(self, raiz: Path | str, nombre: str = "documento",
                 validar_id: Callable[[str], bool] = id_sin_separadores, marca: str | None = None):
        self.raiz = Path(raiz)
        self.raiz.mkdir(parents=True, exist_ok=True)
        self.nombre = nombre
        self._validar_id = validar_id
        self._marca = marca
        self.cerrojo = threading.RLock()

    # ---- rutas -----------------------------------------------------------------------------------

    def carpeta(self, id_documento: str) -> Path:
        """Carpeta del documento (puede no existir). `ValueError` si el id es inválido."""
        if not self._validar_id(id_documento):
            raise ValueError(f"Identificador de {self.nombre} inválido: {id_documento!r}")
        return self.raiz / id_documento

    def ruta_json(self, id_documento: str, nombre: str) -> Path:
        return self.carpeta(id_documento) / f"{nombre}.json"

    def ruta_parquet(self, id_documento: str, nombre: str) -> Path:
        return self.carpeta(id_documento) / f"{nombre}.parquet"

    # ---- ciclo de vida ---------------------------------------------------------------------------

    def crear(self, id_documento: str, reemplazar: bool = False) -> Path:
        """Crea la carpeta del documento (con `reemplazar`, borrando antes lo que hubiera) y la devuelve."""
        carpeta = self.carpeta(id_documento)
        if reemplazar and carpeta.exists():
            shutil.rmtree(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        return carpeta

    def existe(self, id_documento: str) -> bool:
        """True si el documento existe: tiene su `marca` o, sin marca, su carpeta."""
        carpeta = self.carpeta(id_documento)
        return (carpeta / f"{self._marca}.json").exists() if self._marca else carpeta.is_dir()

    def hay_carpeta(self, id_documento: str) -> bool:
        """True si existe la carpeta, aunque esté a medias (sin marca)."""
        return self.carpeta(id_documento).exists()

    def ids(self) -> list[str]:
        """Ids de los documentos, ordenados (solo carpetas; con `marca`, las que la tienen)."""
        return sorted(c.name for c in self.raiz.iterdir()
                      if c.is_dir() and (self._marca is None or (c / f"{self._marca}.json").exists()))

    def borrar(self, id_documento: str) -> bool:
        """Elimina la carpeta entera; False si no existía."""
        carpeta = self.carpeta(id_documento)
        if not carpeta.exists():
            return False
        shutil.rmtree(carpeta)
        return True

    # ---- JSON ------------------------------------------------------------------------------------

    def escribir_json(self, id_documento: str, nombre: str, contenido: Any) -> None:
        """Escribe `<nombre>.json` de forma atómica (la carpeta debe existir: ver `crear`)."""
        escribir_json_atomico(self.ruta_json(id_documento, nombre), contenido)

    def leer_json(self, id_documento: str, nombre: str, reintentar_si_falta: bool = False) -> Any:
        """Contenido de `<nombre>.json` (None si no existe). Ver `leer_json` para `reintentar_si_falta`."""
        return leer_json(self.ruta_json(id_documento, nombre), reintentar_si_falta)

    def tiene_json(self, id_documento: str, nombre: str) -> bool:
        return self.ruta_json(id_documento, nombre).exists()

    def nombres_json(self, id_documento: str) -> list[str]:
        """Nombres (sin extensión, ordenados) de los JSON de la carpeta del documento."""
        return sorted(p.stem for p in self.carpeta(id_documento).glob("*.json"))

    # ---- Parquet ---------------------------------------------------------------------------------

    def escribir_parquet(self, id_documento: str, nombre: str, tabla: pd.DataFrame, index: bool = False) -> None:
        """Escribe `<nombre>.parquet` de forma atómica (la carpeta debe existir)."""
        escribir_parquet_atomico(tabla, self.ruta_parquet(id_documento, nombre), index=index)

    def leer_parquet(self, id_documento: str, nombre: str) -> pd.DataFrame | None:
        """DataFrame de `<nombre>.parquet`; None si no existe."""
        ruta = self.ruta_parquet(id_documento, nombre)
        return pd.read_parquet(ruta) if ruta.exists() else None

    def tiene_parquet(self, id_documento: str, nombre: str) -> bool:
        return self.ruta_parquet(id_documento, nombre).exists()


# ---- archivos planos -----------------------------------------------------------------------------


class ColeccionJson:
    """
    Documentos de un solo archivo: `<raiz>/<id>.json` (sin carpeta por documento), p. ej. las comparativas.

    Args:
        raiz: carpeta de la colección (se crea si no existe).
        prefijo: los ids empiezan por él (`cmp-`); `ids` solo lista los archivos que lo cumplen.
    """

    def __init__(self, raiz: Path | str, prefijo: str = ""):
        self.raiz = Path(raiz)
        self.raiz.mkdir(parents=True, exist_ok=True)
        self.prefijo = prefijo

    def escribir(self, id_documento: str, contenido: Any) -> None:
        """Escritura atómica de `<id>.json`."""
        escribir_json_atomico(self.raiz / f"{id_documento}.json", contenido)

    def leer(self, id_documento: str) -> Any:
        """Contenido de `<id>.json` (None si no existe)."""
        return leer_json(self.raiz / f"{id_documento}.json")

    def ids(self) -> list[str]:
        return sorted(p.stem for p in self.raiz.glob(f"{self.prefijo}*.json"))

    def listar(self) -> list[Any]:
        """Todos los documentos legibles (los que desaparecen entre medias se omiten)."""
        return [documento for i in self.ids() if (documento := self.leer(i)) is not None]
