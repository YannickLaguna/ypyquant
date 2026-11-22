"""
Recalcula `metricas.json` de los experimentos guardados sin repetir ningún backtest.

Responsabilidad:
    Para cada experimento `hecho` del almacén (o los indicados) vuelve a calcular las métricas con
    `evaluacion.reevaluar` a partir de sus Parquet (equity, pesos, operaciones, rotaciones) y reescribe
    `metricas.json` conservando los avisos. Útil cuando cambia una fórmula de métrica o se añade una nueva.
    Con `--simular` no escribe nada y solo muestra qué métricas cambiarían. Omite las importaciones del TFM.

Uso:
    uv run python scripts/reevaluar_experimentos.py [--raiz CARPETA] [--simular] [id ...]
    (por defecto `--raiz data/experimentos`; haz antes una copia de esa carpeta si quieres probar.)

Se relaciona con:
    ypyquant.estrategias.almacen (`AlmacenExperimentos.reevaluar`) y .evaluacion (`reevaluar`).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from ypyquant.estrategias.almacen import AlmacenExperimentos


def _difieren(a, b) -> bool:
    """True si dos valores de métrica son distintos (los floats con tolerancia relativa de 1e-9)."""
    if isinstance(a, float) and isinstance(b, float):
        return not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a != b


def main(argumentos: list[str] | None = None) -> int:
    """Recorre los experimentos y devuelve 0 si todos se reevaluaron (o se omitieron), 1 si alguno falló."""
    parser = argparse.ArgumentParser(description="Recalcula metricas.json de los experimentos guardados.")
    parser.add_argument("ids", nargs="*", help="identificadores a reevaluar (por defecto, todos los guardados)")
    parser.add_argument("--raiz", type=Path, default=None, help="carpeta del almacén (por defecto data/experimentos)")
    parser.add_argument("--simular", action="store_true", help="no escribe: solo informa de las métricas que cambiarían")
    args = parser.parse_args(argumentos)
    almacen = AlmacenExperimentos(args.raiz)
    ids = args.ids or sorted(c.name for c in almacen.raiz.iterdir() if c.is_dir())
    fallos = 0
    for id_experimento in ids:
        try:
            anteriores = almacen.leer_metricas(id_experimento) or {}
            nuevas = almacen.reevaluar(id_experimento, guardar=not args.simular)
        except ValueError as error:
            print(f"OMITIDO  {id_experimento}: {error}")
            continue
        except Exception as error:  # noqa: BLE001 - un experimento roto no debe parar al resto
            fallos += 1
            print(f"FALLO    {id_experimento}: {type(error).__name__}: {error}")
            continue
        claves = (k for k in nuevas if k not in ("validacion", "reevaluacion"))
        cambiadas = sorted(k for k in claves if _difieren(nuevas[k], anteriores.get(k)))
        print(f"{'SIMULADO' if args.simular else 'OK      '} {id_experimento}: {len(cambiadas)} métricas distintas"
              + (f" ({', '.join(cambiadas)})" if cambiadas else ""))
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
