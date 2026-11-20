"""
importar_tfm.py: importa los resultados originales del TFM (rejilla_v2) al almacén de experimentos de la app.

Responsabilidad: CLI fina sobre `ypyquant.estrategias.importacion_tfm`. Lee los Parquet del TFM (solo
lectura, nunca se modifican) y escribe en el almacén `data/experimentos/` experimentos «TFM» que se ven en
Experimentos y se comparan en Comparar con iteraciones posteriores. NO recalcula el TFM.

Uso (desde la raíz del repo):
    uv run python scripts/importar_tfm.py                       # 3 celdas del eje temporal x 6 estrategias = 18
    uv run python scripts/importar_tfm.py --celdas todas        # las 20 celdas (N, T): 120 experimentos
    uv run python scripts/importar_tfm.py --celda 80,36 --celda 20,12
    uv run python scripts/importar_tfm.py --replica 3 --forzar  # réplica 3 en vez de la media de 100
    uv run python scripts/importar_tfm.py --almacen RUTA        # otro almacén (p. ej. una carpeta temporal)

Reimportar un experimento existente lo omite salvo con `--forzar` (idempotente).
Benchmark SPY: se carga con el cargador de precios del repo (caché Parquet + yfinance, que puede
descargar y ampliar `data/cache_precios`); `--sin-benchmark` lo evita y `--cache-precios RUTA` usa otra caché.

Se relaciona con: src/ypyquant/estrategias/importacion_tfm.py (lógica), almacen.py (destino),
precios.py (cargador del benchmark), tests/test_importar_tfm.py y docs/ui.md (sección «Experimentos TFM»).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.importacion_tfm import (
    CELDAS_POR_DEFECTO,
    ESTRATEGIAS_TFM,
    ErrorImportacionTfm,
    celdas_disponibles,
    importar_experimentos_tfm,
)
from ypyquant.estrategias.precios import CargadorPreciosCache

RAIZ_TFM = Path(os.environ.get("TFM_DIR", "C:/Users/yanni/Documents/Codigo/Proyectos/tfm"))
CARPETA_TFM_POR_DEFECTO = RAIZ_TFM / "data" / "reportes" / "rejilla_v2"


def _celda(texto: str) -> tuple[int, int]:
    """Convierte `N,T` en `(N, T)` (argumento de `--celda`)."""
    try:
        n, t = (int(x) for x in texto.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"celda inválida {texto!r}: usa N,T (p. ej. 80,36)") from error
    return n, t


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Importa los resultados originales del TFM como experimentos de solo lectura.")
    parser.add_argument("--origen", type=Path, default=CARPETA_TFM_POR_DEFECTO,
                        help="carpeta rejilla_v2 del TFM (por defecto %(default)s)")
    parser.add_argument("--almacen", type=Path, default=None,
                        help="carpeta del almacén de experimentos (por defecto data/experimentos/ del repo)")
    parser.add_argument("--celdas", choices=["defecto", "todas"], default="defecto",
                        help="defecto = (20,12), (80,36), (150,120); todas = las 20 celdas del TFM")
    parser.add_argument("--celda", type=_celda, action="append", metavar="N,T",
                        help="celda concreta (repetible); sustituye a --celdas")
    parser.add_argument("--estrategia", action="append", choices=ESTRATEGIAS_TFM,
                        help="limita a estas estrategias del TFM (repetible; por defecto las 6)")
    parser.add_argument("--replica", type=int, default=None, metavar="b",
                        help="réplica concreta (0-99) en vez de la serie promediada; trae pesos si b < 20")
    parser.add_argument("--forzar", action="store_true", help="sobrescribe los experimentos ya importados")
    parser.add_argument("--sin-benchmark", action="store_true", help="no carga SPY (sin alfa/beta ni benchmark)")
    parser.add_argument("--cache-precios", type=Path, default=None, help="carpeta de caché de precios (por defecto data/cache_precios)")
    return parser


def main(argv: list[str] | None = None) -> int:
    argumentos = crear_parser().parse_args(argv)
    try:
        if argumentos.celda:
            celdas = argumentos.celda
        elif argumentos.celdas == "todas":
            celdas = celdas_disponibles(argumentos.origen)
        else:
            celdas = list(CELDAS_POR_DEFECTO)
        cargador = None if argumentos.sin_benchmark else CargadorPreciosCache(argumentos.cache_precios)
        almacen = AlmacenExperimentos(argumentos.almacen)
        acciones = importar_experimentos_tfm(
            argumentos.origen, almacen, celdas=celdas, estrategias=argumentos.estrategia, replica=argumentos.replica,
            forzar=argumentos.forzar, cargador_precios=cargador)
    except ErrorImportacionTfm as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    for accion in acciones:
        print(f"{accion['accion']:<11} {accion['id']}")
    resumen = {k: sum(a["accion"] == k for a in acciones) for k in ("importado", "sobrescrito", "omitido")}
    print(f"\nAlmacén: {almacen.raiz}\n" + " · ".join(f"{v} {k}" for k, v in resumen.items()))
    if resumen["omitido"]:
        print("Los omitidos ya existían; usa --forzar para reimportarlos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
