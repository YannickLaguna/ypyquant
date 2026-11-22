"""
Raíz de datos: la única que sabe cómo cuelgan los almacenes de la carpeta `data/`.

Responsabilidad: construir, a partir de una carpeta base, las rutas de los almacenes hermanos (experimentos, barridos,
configuraciones, marcos, paper interno) y del archivo común del periodo reservado, para que nadie las recomponga con `parent / "..."`.
El paper interno cuelga de `paper_interno` y el de Alpaca (por perfil) de `paper_trading`; los almacenes nuevos se añaden aquí como propiedad.

Se relaciona con:
- ypyquant.almacen_documentos (los almacenes que se construyen sobre estas rutas) y los adaptadores en
  ypyquant.estrategias (`AlmacenExperimentos`, `AlmacenBarridos`, `AlmacenConfiguraciones`, `AlmacenMarcos`).
- app/api/main.py y app/api/contexto.py (construyen los almacenes de la API desde una `RaizDatos`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RaizDatos:
    """
    Carpeta base de los datos (por defecto `data/` del repo) y rutas derivadas.

    `nombre_experimentos` existe porque los tests y scripts eligen la carpeta de experimentos con otro nombre
    (`de_experimentos`): el resto de almacenes cuelga siempre de la misma base.
    """

    base: Path = RAIZ_REPO / "data"
    nombre_experimentos: str = "experimentos"

    @classmethod
    def de_experimentos(cls, raiz_experimentos: Path | str) -> RaizDatos:
        """Raíz cuya carpeta de experimentos es `raiz_experimentos`; los demás almacenes son hermanos suyos."""
        ruta = Path(raiz_experimentos)
        return cls(base=ruta.parent, nombre_experimentos=ruta.name)

    @property
    def experimentos(self) -> Path:
        return self.base / self.nombre_experimentos

    @property
    def barridos(self) -> Path:
        return self.base / "barridos"

    @property
    def configuraciones(self) -> Path:
        return self.base / "configuraciones"

    @property
    def marcos(self) -> Path:
        return self.base / "marcos"

    @property
    def paper_trading(self) -> Path:
        """Carpeta del paper trading con Alpaca: una subcarpeta por perfil (`<PERFIL>/vinculo.json`, `bot.json`, ...)."""
        return self.base / "paper_trading"

    @property
    def paper_interno(self) -> Path:
        """Sesiones del paper interno: una carpeta por sesión (ver `ypyquant.paper.almacen`)."""
        return self.base / "paper_interno"

    @property
    def reservado(self) -> Path:
        """Archivo con el contador de usos del periodo reservado (común a todos los barridos)."""
        return self.base / "reservado.json"
