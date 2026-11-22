"""
Índice en memoria «receta -> experimentos», para responder «backtests de esta configuración» sin recorrer el disco.

Responsabilidad:
- `IndiceRecetas`: `{huella_receta: {id_experimento}}` y su inverso. Se construye UNA vez leyendo los `config.json`
  del almacén (`construir`) y se mantiene en memoria con `anadir` / `quitar`, que llama el propio `AlmacenExperimentos`
  al registrar y al borrar. Es seguro entre hilos.
- Decisión (memoria frente a fichero índice): un fichero duplicaría la verdad del disco y habría que mantenerlo
  coherente ante escrituras concurrentes, borrados manuales y sincronización; la construcción cuesta una lectura de
  cada `config.json` (decenas de milisegundos para miles de experimentos) y se paga una sola vez por proceso.
  Si el almacén cambiara por fuera de este proceso (borrado o copia manual de carpetas), `AlmacenExperimentos.indexar()`
  lo reconstruye; las consultas, además, descartan los ids cuya carpeta ya no existe.
- Las configuraciones que no validan como `ConfiguracionExperimento` (importadas del TFM, dañadas) no se indexan.

Se relaciona con:
- ypyquant.estrategias.almacen (`AlmacenExperimentos` lo posee y lo expone con `resumenes_de_receta`) y .receta
  (la huella de receta es la clave); .servicio_experimentos y app/api/rutas_configuraciones.py lo consultan a través del almacén.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from ypyquant.estrategias.experimentos import ConfiguracionExperimento


def huella_receta_de(config: Mapping[str, Any]) -> str | None:
    """Huella de receta de un `config.json` guardado; None si no es una configuración de experimento válida."""
    try:
        return ConfiguracionExperimento(**config).huella_receta()
    except (ValidationError, TypeError, KeyError, ValueError):
        return None


class IndiceRecetas:
    """Mapa receta -> ids de experimento, construido bajo demanda y mantenido por el almacén."""

    def __init__(self) -> None:
        self._por_receta: dict[str, set[str]] = {}
        self._receta_de: dict[str, str] = {}
        self._cerrojo = threading.RLock()
        self.construido = False

    def construir(self, configs: Iterable[tuple[str, Mapping[str, Any]]]) -> None:
        """Reemplaza el contenido por `(id, config.json)` de cada experimento del disco."""
        with self._cerrojo:
            self._por_receta.clear()
            self._receta_de.clear()
            for id_experimento, config in configs:
                huella = huella_receta_de(config)
                if huella is not None:
                    self._indexar(id_experimento, huella)
            self.construido = True

    def _indexar(self, id_experimento: str, huella: str) -> None:
        self._por_receta.setdefault(huella, set()).add(id_experimento)
        self._receta_de[id_experimento] = huella

    def anadir(self, id_experimento: str, config: Mapping[str, Any]) -> None:
        """Indexa (o reindexa) un experimento recién registrado."""
        huella = huella_receta_de(config)
        with self._cerrojo:
            self.quitar(id_experimento)
            if huella is not None:
                self._indexar(id_experimento, huella)

    def quitar(self, id_experimento: str) -> None:
        """Olvida un experimento borrado (no falla si no estaba)."""
        with self._cerrojo:
            huella = self._receta_de.pop(id_experimento, None)
            if huella is not None:
                ids = self._por_receta.get(huella, set())
                ids.discard(id_experimento)
                if not ids:
                    self._por_receta.pop(huella, None)

    def ids(self, huella_receta: str) -> list[str]:
        """Ids de los experimentos de esa receta (orden estable por id)."""
        with self._cerrojo:
            return sorted(self._por_receta.get(huella_receta, ()))
