"""
exportar_membresias_sp500.py — Genera `universos/sp500_membresias.json` desde la base del TFM.

Responsabilidad: leer en SOLO LECTURA `tickers.index_memberships` de `TFM.db` (la misma fuente que usa la
prueba de paridad del motor de rejilla) y volcar los intervalos de pertenencia al S&P 500 de cada ticker
a un JSON versionado en el repo, para que el universo point-in-time `sp500_pit` no dependa de la ruta del TFM.

Uso:
    uv run python scripts/exportar_membresias_sp500.py                  # TFM_DIR o la ruta por defecto
    uv run python scripts/exportar_membresias_sp500.py --db RUTA/TFM.db --salida universos/sp500_membresias.json

Se relaciona con: ypyquant.data.universos (`extraer_membresias_desde_sqlite`,
`cargar_membresias_sp500`) y universos/sp500_membresias.json.
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

from ypyquant.data.universos import RUTA_MEMBRESIAS, extraer_membresias_desde_sqlite

RUTA_TFM_POR_DEFECTO = "C:/Users/yanni/Documents/Codigo/Proyectos/tfm"


def main() -> None:
    analizador = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    raiz_tfm = Path(os.environ.get("TFM_DIR", RUTA_TFM_POR_DEFECTO))
    analizador.add_argument("--db", default=str(raiz_tfm / "data" / "TFM.db"), help="ruta a TFM.db (solo lectura)")
    analizador.add_argument("--salida", default=str(RUTA_MEMBRESIAS), help="JSON de salida")
    argumentos = analizador.parse_args()

    miembros = extraer_membresias_desde_sqlite(argumentos.db)
    salida = {
        "indice": "SP500",
        "fuente": "TFM.db, tabla tickers, columna index_memberships (solo entradas del índice SP500)",
        "extraido": date.today().isoformat(),
        "nota": ("Cobertura fiable desde ~2000 (480-520 miembros por fecha); antes de 1999 las membresías son "
                 "incompletas. Tickers en formato yfinance (BRK-B). Una empresa que cambió de ticker figura con el vigente."),
        "n_tickers": len(miembros),
        "miembros": miembros,
    }
    destino = Path(argumentos.salida)
    destino.write_text(json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(miembros)} tickers con membresía del S&P 500 -> {destino}")


if __name__ == "__main__":
    main()
