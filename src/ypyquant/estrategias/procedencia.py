"""
Procedencia de un experimento: con qué código, versiones y datos se obtuvo el resultado.

Responsabilidad:
- `obtener_procedencia(precios, benchmark)`: dict serializable con el commit git y si el árbol
  estaba sucio, la versión de ypyquant, Python, numpy/pandas/cvxpy, un hash SHA-256 estable de
  los precios usados y la fecha de ejecución.
- `commit_git()`: `{commit, sucio}` del repo donde vive el paquete, calculado UNA vez por proceso
  (el estado «sucio» es el del primer uso; solo cuenta cambios en archivos versionados). Si el
  paquete no está en un repo git (instalado en otra app) devuelve `{None, None}` sin fallar.
- `hash_precios(precios, benchmark)`: hash independiente del orden de los tickers.

La procedencia NO entra en la huella del experimento (misma configuración = mismo id aunque se
ejecute otro día o con otro commit); sirve para auditar y reproducir.

Se relaciona con:
- ypyquant.estrategias.experimentos (la calcula en `ejecutar_con_datos`) y .almacen (la guarda en
  `procedencia.json` y la devuelve en `leer`).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parents[3]


def _git(*argumentos: str) -> str | None:
    """Ejecuta git en la raíz del repo; None si git no existe, falla o tarda más de 5 s."""
    try:
        salida = subprocess.run(["git", *argumentos], cwd=RAIZ_REPO, capture_output=True, text=True, timeout=5,
                                check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return salida.stdout.strip() if salida.returncode == 0 else None


@lru_cache(maxsize=1)
def commit_git() -> dict[str, Any]:
    """`{commit, sucio}`; ambos None si el paquete no está dentro de un repo git. Cacheado por proceso."""
    if not (RAIZ_REPO / ".git").exists():  # .git es carpeta o archivo (worktree)
        return {"commit": None, "sucio": None}
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "sucio": None}
    estado = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "sucio": None if estado is None else bool(estado)}


def _version(paquete: str) -> str | None:
    try:
        return importlib.metadata.version(paquete)
    except importlib.metadata.PackageNotFoundError:
        return None


def hash_precios(precios: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None = None) -> str:
    """
    SHA-256 de los precios: por ticker ordenado, su nombre, columnas y `hash_pandas_object` (índice y valores).

    El benchmark entra con la marca `__benchmark__`. Mismos datos = mismo hash, sea cual sea el orden del dict.
    """
    resumen = hashlib.sha256()
    elementos = sorted(precios.items())
    if benchmark is not None:
        elementos.append(("__benchmark__", benchmark))
    for ticker, df in elementos:
        resumen.update(ticker.encode("utf-8"))
        resumen.update(",".join(map(str, df.columns)).encode("utf-8"))
        resumen.update(np.ascontiguousarray(pd.util.hash_pandas_object(df, index=True).to_numpy()).tobytes())
    return resumen.hexdigest()


def obtener_procedencia(precios: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None = None) -> dict[str, Any]:
    """Procedencia completa (JSON-serializable) de una ejecución con esos precios."""
    git = commit_git()
    return {
        "commit": git["commit"],
        "arbol_sucio": git["sucio"],
        "version_ypyquant": _version("ypyquant"),
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "cvxpy": _version("cvxpy"),
        "hash_precios": hash_precios(precios, benchmark),
        "ejecutado": datetime.now().isoformat(timespec="seconds"),
    }
