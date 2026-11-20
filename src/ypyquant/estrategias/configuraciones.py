"""
Configuraciones guardadas: recetas congeladas y versionadas de una estrategia lista para relanzar.

Responsabilidad:
- `ConfiguracionGuardada` (Pydantic): estrategia + parámetros + universo + frecuencia + costes + benchmark +
  semilla, SIN fechas (las fechas pertenecen al backtest, no a la receta). `a_experimento(inicio, fin)` construye
  una `ConfiguracionExperimento` normal (misma huella que el experimento de origen si coinciden fechas y
  capital: se reutilizan los resultados ya calculados). `huella_estrategia` = `huella_receta()` de esa
  configuración (sin fechas ni capital). `id_marco` (opcional) enlaza con un `MarcoComparacion` (ver `marcos.py`).
- `OrigenConfiguracion`: de dónde sale (`experimento` | `barrido` | `manual`), criterio de elección en un barrido
  (`mejor` | `meseta` | `elegida`) y las métricas EN EL MOMENTO DE GUARDAR (para medir luego la degradación).
- `AlmacenConfiguraciones`: persistencia en `data/configuraciones/<id>/v<n>.json` + `indice.json` (archivada,
  última versión). Cada versión es inmutable: editar = `nueva_version`. Escritura atómica con los mismos
  ayudantes que `AlmacenExperimentos`.
- Universo: un nombre de archivo (`sp500:30`) se CONGELA como lista de tickers (la etiqueta conserva el nombre);
  `sp500_pit` es dinámico por definición y se guarda como nombre.

Se relaciona con:
- ypyquant.estrategias.experimentos (ConfiguracionExperimento, resolver_universo) y .almacen (E/S JSON atómica).
- ypyquant.estrategias.marcos (el `id_marco` de una configuración).
- app/api/rutas_configuraciones.py (las expone) y app/web (página Configuraciones).
"""

from __future__ import annotations

import re
import threading
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ypyquant.estrategias.almacen import _escribir_json, _leer_json
from ypyquant.estrategias.experimentos import (
    ConfiguracionExperimento,
    Frecuencia,
    es_universo_pit,
    resolver_universo,
)

RAIZ_REPO = Path(__file__).resolve().parents[3]
RAIZ_POR_DEFECTO = RAIZ_REPO / "data" / "configuraciones"
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


class ConfiguracionGuardada(BaseModel):
    """
    Receta congelada de una estrategia (ver cabecera del módulo). `version` empieza en 1; `deriva_de` es la
    versión de la que se creó (None en la primera). `universo` es una lista de tickers o el nombre `sp500_pit`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    version: int = Field(1, ge=1)
    nombre: str = Field(min_length=1)
    notas: str = ""
    estrategia: str
    parametros: dict[str, Any] = Field(default_factory=dict)
    universo: list[str] | str
    universo_etiqueta: str | None = None
    frecuencia_rebalanceo: Frecuencia
    comision: float = Field(0.001, ge=0, le=0.1)
    slippage: float = Field(0.001, ge=0, le=0.1)
    benchmark: str = "SPY"
    capital_inicial: float = Field(100_000.0, gt=0)
    semilla: int = 42
    dias_historia_previa: int = Field(1100, ge=0, le=7300)
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
        universo = config.universo
        etiqueta = None
        if isinstance(universo, str):
            etiqueta = universo
            if not es_universo_pit(universo):
                universo = resolver_universo(universo)
        guardada = cls(
            id=id or crear_slug(nombre), nombre=nombre, notas=notas, estrategia=config.estrategia,
            parametros=config.parametros, universo=universo, universo_etiqueta=etiqueta,
            frecuencia_rebalanceo=config.frecuencia_rebalanceo, comision=config.comision, slippage=config.slippage,
            benchmark=config.benchmark, capital_inicial=config.capital_inicial, semilla=config.semilla,
            dias_historia_previa=config.dias_historia_previa, origen=origen or OrigenConfiguracion(),
            creada=datetime.now().isoformat(timespec="seconds"), etiquetas=etiquetas or [], id_marco=id_marco,
        )
        guardada.huella_estrategia = guardada.a_experimento("2000-01-01", "2000-01-02").huella_receta()
        return guardada

    def a_experimento(self, fecha_inicio: str, fecha_fin: str, capital_inicial: float | None = None) -> ConfiguracionExperimento:
        """Configuración de backtest de esta receta en `[fecha_inicio, fecha_fin]` (capital: el de la receta por defecto)."""
        return ConfiguracionExperimento(
            estrategia=self.estrategia, parametros=self.parametros, universo=self.universo,
            fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, capital_inicial=capital_inicial or self.capital_inicial,
            comision=self.comision, slippage=self.slippage, frecuencia_rebalanceo=self.frecuencia_rebalanceo,
            benchmark=self.benchmark, semilla=self.semilla, dias_historia_previa=self.dias_historia_previa,
        ).normalizada()


class AlmacenConfiguraciones:
    """
    Persistencia de configuraciones en `<raiz>/<id>/v<n>.json` + `<raiz>/<id>/indice.json`
    (`{"ultima_version", "archivada"}`). Las versiones nunca se reescriben.
    """

    def __init__(self, raiz: Path | str | None = None):
        self.raiz = Path(raiz) if raiz else RAIZ_POR_DEFECTO
        self.raiz.mkdir(parents=True, exist_ok=True)
        self._cerrojo = threading.Lock()

    def _carpeta(self, id_configuracion: str) -> Path:
        if not id_configuracion or not re.fullmatch(r"[a-z0-9][a-z0-9\-_.]*", id_configuracion) or ".." in id_configuracion:
            raise ValueError(f"Identificador de configuración inválido: {id_configuracion!r}")
        return self.raiz / id_configuracion

    def _indice(self, carpeta: Path) -> dict[str, Any]:
        indice = _leer_json(carpeta / "indice.json")
        if indice is None:
            raise KeyError(carpeta.name)
        return indice

    def guardar(self, configuracion: ConfiguracionGuardada) -> ConfiguracionGuardada:
        """Guarda la versión 1. `ConfiguracionExistente` si el id ya existe."""
        carpeta = self._carpeta(configuracion.id)
        with self._cerrojo:
            if (carpeta / "indice.json").exists():
                raise ConfiguracionExistente(f"Ya existe la configuración '{configuracion.id}'")
            carpeta.mkdir(parents=True, exist_ok=True)
            _escribir_json(carpeta / "v1.json", configuracion.model_dump())
            _escribir_json(carpeta / "indice.json", {"ultima_version": 1, "archivada": False})
        return configuracion

    def leer(self, id_configuracion: str, version: int | None = None) -> ConfiguracionGuardada:
        """Una versión (la última por defecto); incluye `archivada` del índice. `KeyError` si no existe."""
        carpeta = self._carpeta(id_configuracion)
        indice = self._indice(carpeta)
        numero = version or indice["ultima_version"]
        datos = _leer_json(carpeta / f"v{numero}.json")
        if datos is None:
            raise KeyError(f"{id_configuracion} v{numero}")
        return ConfiguracionGuardada(**{**datos, "archivada": indice["archivada"]})

    def versiones(self, id_configuracion: str) -> list[ConfiguracionGuardada]:
        """Todas las versiones, de la más antigua a la última."""
        ultima = self._indice(self._carpeta(id_configuracion))["ultima_version"]
        return [self.leer(id_configuracion, n) for n in range(1, ultima + 1)]

    def listar(self, estrategia: str | None = None, texto: str | None = None,
               archivadas: bool = False) -> list[ConfiguracionGuardada]:
        """Última versión de cada configuración (más recientes primero), filtrable por estrategia y texto libre."""
        resultado = []
        for carpeta in self.raiz.iterdir():
            if not carpeta.is_dir() or not (carpeta / "indice.json").exists():
                continue
            try:
                configuracion = self.leer(carpeta.name)
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
        carpeta = self._carpeta(id_configuracion)
        with self._cerrojo:
            actual = self.leer(id_configuracion)
            datos = {**actual.model_dump(), **cambios}
            if "universo" in cambios and isinstance(cambios["universo"], str) and not es_universo_pit(cambios["universo"]):
                datos["universo_etiqueta"] = cambios["universo"]
                datos["universo"] = resolver_universo(cambios["universo"])
            datos.update(version=actual.version + 1, deriva_de=actual.version, origen=OrigenConfiguracion(),
                         creada=datetime.now().isoformat(timespec="seconds"))
            nueva = ConfiguracionGuardada(**datos)
            experimento = nueva.a_experimento("2000-01-01", "2000-01-02")  # valida parámetros y universo
            nueva.parametros = experimento.parametros
            nueva.huella_estrategia = experimento.huella_receta()
            _escribir_json(carpeta / f"v{nueva.version}.json", nueva.model_dump())
            _escribir_json(carpeta / "indice.json", {"ultima_version": nueva.version, "archivada": actual.archivada})
        return nueva

    def archivar(self, id_configuracion: str, archivada: bool = True) -> None:
        """Marca (o desmarca) como archivada: desaparece del listado por defecto pero conserva todo."""
        carpeta = self._carpeta(id_configuracion)
        with self._cerrojo:
            indice = self._indice(carpeta)
            _escribir_json(carpeta / "indice.json", {**indice, "archivada": archivada})

    def borrar(self, id_configuracion: str) -> bool:
        """Borra la configuración con todas sus versiones. False si no existía."""
        import shutil

        carpeta = self._carpeta(id_configuracion)
        with self._cerrojo:
            if not carpeta.exists():
                return False
            shutil.rmtree(carpeta)
        return True
