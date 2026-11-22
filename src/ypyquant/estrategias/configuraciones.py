"""
Configuraciones guardadas: recetas congeladas y versionadas de una estrategia lista para relanzar.

Responsabilidad:
- `ConfiguracionGuardada` (Pydantic): ES una `Receta` (ver `receta.py`: estrategia + parámetros + entorno, SIN fechas,
  porque las fechas pertenecen al backtest, no a la receta) más identidad, notas, versión, capital por defecto y
  origen. `receta()` da la receta pura; `a_experimento(inicio, fin)` la compone con un periodo
  (`ConfiguracionExperimento.desde_receta`: misma huella que el experimento de origen si coinciden fechas y capital,
  así se reutilizan los resultados ya calculados). `huella_estrategia` = `receta().huella()`, sin fechas falsas.
  `id_marco` (opcional) enlaza con un `MarcoComparacion` (ver `marcos.py`).
- `OrigenConfiguracion`: de dónde sale (`experimento` | `barrido` | `manual`), criterio de elección en un barrido
  (`mejor` | `meseta` | `elegida`) y las métricas EN EL MOMENTO DE GUARDAR (para medir luego la degradación).
- `AlmacenConfiguraciones`: persistencia en `data/configuraciones/<id>/v<n>.json` + `indice.json` (archivada,
  última versión). Cada versión es inmutable: editar = `nueva_version`. Escritura atómica con los mismos
  ayudantes que `AlmacenExperimentos`.
- Universo: un nombre de archivo (`sp500:30`) se CONGELA como lista de tickers (la etiqueta conserva el nombre);
  `sp500_pit` es dinámico por definición y se guarda como nombre.

Se relaciona con:
- ypyquant.estrategias.receta (`Receta`), .experimentos (ConfiguracionExperimento, resolver_universo) y .almacen (E/S JSON atómica).
- ypyquant.estrategias.marcos (el `id_marco` de una configuración).
- app/api/rutas_configuraciones.py (las expone) y app/web (página Configuraciones).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ypyquant.almacen_documentos import AlmacenDocumentos, id_slug
from ypyquant.estrategias.experimentos import (
    ConfiguracionExperimento,
    es_universo_pit,
    resolver_universo,
)
from ypyquant.estrategias.receta import Frecuencia, Receta
from ypyquant.raiz_datos import RaizDatos

# Campos que `nueva_version` permite cambiar (la estrategia no: otra estrategia es otra configuración)
CAMPOS_EDITABLES = ("nombre", "notas", "etiquetas", "parametros", "universo", "frecuencia_rebalanceo", "comision",
                    "slippage", "benchmark", "capital_inicial", "semilla", "dias_historia_previa", "id_marco")


class ConfiguracionExistente(ValueError):
    """Ya hay una configuración con ese id."""


def crear_slug(nombre: str) -> str:
    """Id estable a partir de un nombre: minúsculas, sin tildes, guiones. `ValueError` si no queda nada."""
    sin_tildes = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", sin_tildes.lower()).strip("-")
    if not slug:
        raise ValueError(f"No se puede crear un id a partir de {nombre!r}")
    return slug


class OrigenConfiguracion(BaseModel):
    """
    De dónde sale una configuración. `metricas` son las del experimento en el momento de guardarla;
    `aviso` es un texto libre para la UI (p. ej. «elegida sobre datos de desarrollo, sin validar en el reservado»).
    """

    model_config = ConfigDict(extra="forbid")

    tipo: Literal["experimento", "barrido", "manual"] = "manual"
    id_experimento: str | None = None
    id_barrido: str | None = None
    criterio: Literal["mejor", "meseta", "elegida"] | None = None
    metricas: dict[str, Any] = Field(default_factory=dict)
    fechas: list[str] | None = None
    aviso: str | None = None


class ConfiguracionGuardada(Receta):
    """
    Receta congelada de una estrategia (ver cabecera del módulo): ES una `Receta` (estrategia, parámetros y entorno
    heredados) más identidad, notas, versión y capital por defecto. `version` empieza en 1; `deriva_de` es la
    versión de la que se creó (None en la primera). `universo` es una lista de tickers o el nombre `sp500_pit`.
    A diferencia de la receta pura, la frecuencia ya está resuelta (obligatoria).
    """

    id: str
    version: int = Field(1, ge=1)
    nombre: str = Field(min_length=1)
    notas: str = ""
    universo_etiqueta: str | None = None
    frecuencia_rebalanceo: Frecuencia  # type: ignore[assignment]  # obligatoria: la receta congelada ya la resolvió
    capital_inicial: float = Field(100_000.0, gt=0)
    origen: OrigenConfiguracion = Field(default_factory=OrigenConfiguracion)
    creada: str = ""
    etiquetas: list[str] = Field(default_factory=list)
    huella_estrategia: str = ""
    archivada: bool = False
    id_marco: str | None = None
    deriva_de: int | None = None

    @classmethod
    def desde_experimento(
        cls, config: ConfiguracionExperimento, nombre: str, notas: str = "", origen: OrigenConfiguracion | None = None,
        id_marco: str | None = None, etiquetas: list[str] | None = None, id: str | None = None,
    ) -> ConfiguracionGuardada:
        """
        Congela la receta de `config` (se normaliza: parámetros completos y frecuencia resuelta). Lanza
        `ValidationError` con parámetros inválidos y `ValueError` si el universo con nombre no existe.
        """
        config = config.normalizada()
        receta = config.receta()
        universo = receta.universo
        etiqueta = None
        if isinstance(universo, str):
            etiqueta = universo
            if not es_universo_pit(universo):
                universo = resolver_universo(universo)
        guardada = cls(
            **{**receta.model_dump(), "universo": universo}, id=id or crear_slug(nombre), nombre=nombre, notas=notas,
            universo_etiqueta=etiqueta, capital_inicial=config.capital_inicial, origen=origen or OrigenConfiguracion(),
            creada=datetime.now().isoformat(timespec="seconds"), etiquetas=etiquetas or [], id_marco=id_marco,
        )
        guardada.huella_estrategia = guardada.receta().huella()
        return guardada

    def a_experimento(self, fecha_inicio: str, fecha_fin: str, capital_inicial: float | None = None) -> ConfiguracionExperimento:
        """Configuración de backtest de esta receta en `[fecha_inicio, fecha_fin]` (capital: el de la receta por defecto)."""
        return ConfiguracionExperimento.desde_receta(
            self.receta(), fecha_inicio, fecha_fin, capital_inicial or self.capital_inicial).normalizada()


class AlmacenConfiguraciones:
    """
    Persistencia de configuraciones en `<raiz>/<id>/v<n>.json` + `<raiz>/<id>/indice.json`
    (`{"ultima_version", "archivada"}`). Las versiones nunca se reescriben.

    Adaptador de dominio sobre un `AlmacenDocumentos` (`self.documentos`, ids tipo slug, `indice.json` como marca): aquí
    solo viven el versionado, el índice y los filtros.
    """

    def __init__(self, raiz: Path | str | None = None):
        self.documentos = AlmacenDocumentos(raiz or RaizDatos().configuraciones, nombre="configuración",
                                            validar_id=id_slug, marca="indice")
        self.raiz = self.documentos.raiz

    def _indice(self, id_configuracion: str) -> dict[str, Any]:
        indice = self.documentos.leer_json(id_configuracion, "indice")
        if indice is None:
            raise KeyError(id_configuracion)
        return indice

    def guardar(self, configuracion: ConfiguracionGuardada) -> ConfiguracionGuardada:
        """Guarda la versión 1. `ConfiguracionExistente` si el id ya existe."""
        with self.documentos.cerrojo:
            if self.documentos.existe(configuracion.id):
                raise ConfiguracionExistente(f"Ya existe la configuración '{configuracion.id}'")
            self.documentos.crear(configuracion.id)
            self.documentos.escribir_json(configuracion.id, "v1", configuracion.model_dump())
            self.documentos.escribir_json(configuracion.id, "indice", {"ultima_version": 1, "archivada": False})
        return configuracion

    def leer(self, id_configuracion: str, version: int | None = None) -> ConfiguracionGuardada:
        """Una versión (la última por defecto); incluye `archivada` del índice. `KeyError` si no existe."""
        indice = self._indice(id_configuracion)
        numero = version or indice["ultima_version"]
        datos = self.documentos.leer_json(id_configuracion, f"v{numero}")
        if datos is None:
            raise KeyError(f"{id_configuracion} v{numero}")
        return ConfiguracionGuardada(**{**datos, "archivada": indice["archivada"]})

    def versiones(self, id_configuracion: str) -> list[ConfiguracionGuardada]:
        """Todas las versiones, de la más antigua a la última."""
        ultima = self._indice(id_configuracion)["ultima_version"]
        return [self.leer(id_configuracion, n) for n in range(1, ultima + 1)]

    def listar(self, estrategia: str | None = None, texto: str | None = None,
               archivadas: bool = False) -> list[ConfiguracionGuardada]:
        """Última versión de cada configuración (más recientes primero), filtrable por estrategia y texto libre."""
        resultado = []
        for id_configuracion in self.documentos.ids():
            try:
                configuracion = self.leer(id_configuracion)
            except (KeyError, ValueError):
                continue
            if configuracion.archivada and not archivadas:
                continue
            if estrategia and configuracion.estrategia != estrategia:
                continue
            if texto and texto.lower() not in " ".join(
                    [configuracion.id, configuracion.nombre, configuracion.notas, *configuracion.etiquetas]).lower():
                continue
            resultado.append(configuracion)
        return sorted(resultado, key=lambda c: c.creada, reverse=True)

    def nueva_version(self, id_configuracion: str, cambios: dict[str, Any]) -> ConfiguracionGuardada:
        """
        Crea la versión n+1 aplicando `cambios` (solo `CAMPOS_EDITABLES`) a la última. Se revalida (parámetros
        incluidos) y se recalcula la huella. `ValueError` si un campo no es editable; `ValidationError` si es inválido.
        """
        no_editables = sorted(set(cambios) - set(CAMPOS_EDITABLES))
        if no_editables:
            raise ValueError(f"campo no editable: {', '.join(no_editables)} (editables: {', '.join(CAMPOS_EDITABLES)})")
        with self.documentos.cerrojo:
            actual = self.leer(id_configuracion)
            datos = {**actual.model_dump(), **cambios}
            if "universo" in cambios and isinstance(cambios["universo"], str) and not es_universo_pit(cambios["universo"]):
                datos["universo_etiqueta"] = cambios["universo"]
                datos["universo"] = resolver_universo(cambios["universo"])
            datos.update(version=actual.version + 1, deriva_de=actual.version, origen=OrigenConfiguracion(),
                         creada=datetime.now().isoformat(timespec="seconds"))
            nueva = ConfiguracionGuardada(**datos)
            receta = nueva.receta().normalizada()  # valida parámetros y universo
            nueva.parametros = receta.parametros
            nueva.huella_estrategia = receta.huella()
            self.documentos.escribir_json(id_configuracion, f"v{nueva.version}", nueva.model_dump())
            self.documentos.escribir_json(id_configuracion, "indice",
                                          {"ultima_version": nueva.version, "archivada": actual.archivada})
        return nueva

    def archivar(self, id_configuracion: str, archivada: bool = True) -> None:
        """Marca (o desmarca) como archivada: desaparece del listado por defecto pero conserva todo."""
        with self.documentos.cerrojo:
            indice = self._indice(id_configuracion)
            self.documentos.escribir_json(id_configuracion, "indice", {**indice, "archivada": archivada})

    def borrar(self, id_configuracion: str) -> bool:
        """Borra la configuración con todas sus versiones. False si no existía."""
        with self.documentos.cerrojo:
            return self.documentos.borrar(id_configuracion)
