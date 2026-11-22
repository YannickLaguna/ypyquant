"""
__main__.py — CLI del paper interno: `python -m ypyquant.paper <orden>` (la usa el Programador de tareas de Windows).

Responsabilidad:
    Órdenes (todas idempotentes; la raíz de las sesiones es `data/paper_interno/` o la de `YPYQUANT_PAPER_RAIZ`):
    - `avanzar [--sesion ID] [--hasta AAAA-MM-DD]`: procesa los días cerrados pendientes de las sesiones activas (o de
      una) con una descarga por ticker. Imprime un resumen por sesión y sale con código 1 si alguna falló (para que el
      programador lo registre) y 0 en otro caso (también si no había nada pendiente).
    - `estado`: una línea por sesión (estado, último día, retorno, días pendientes).
    - `crear --experimento ID [--modo cierre|apertura_siguiente] [--capital N] [--nombre T] [--comision X] [--slippage X]
      [--fecha-alta AAAA-MM-DD]`: congela un experimento terminado en una sesión nueva.
    - `pausar|reanudar|detener ID`.
    Los precios salen de `fuente_por_defecto()` (`YPYQUANT_PRECIOS=sintetico` para pruebas sin red).

Se relaciona con:
    - ypyquant.paper.servicio (`ServicioPaper`), scripts/instalar_tarea_paper.ps1 (tarea diaria que lo invoca) y
      tests/test_paper_cli.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from ypyquant.data.fuentes import fuente_por_defecto
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.paper.almacen import AlmacenPaper
from ypyquant.paper.servicio import ServicioPaper, receta_de_experimento
from ypyquant.raiz_datos import RaizDatos


def _servicio() -> ServicioPaper:
    raiz = Path(os.environ["YPYQUANT_PAPER_RAIZ"]) if os.environ.get("YPYQUANT_PAPER_RAIZ") else RaizDatos().paper_interno
    return ServicioPaper(AlmacenPaper(raiz), fuente_por_defecto())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ypyquant.paper", description="Paper trading interno")
    ordenes = parser.add_subparsers(dest="orden", required=True)
    avanzar = ordenes.add_parser("avanzar", help="procesa los días cerrados pendientes")
    avanzar.add_argument("--sesion")
    avanzar.add_argument("--hasta", type=date.fromisoformat)
    ordenes.add_parser("estado", help="resumen de todas las sesiones")
    crear = ordenes.add_parser("crear", help="crea una sesión desde un experimento terminado")
    crear.add_argument("--experimento", required=True)
    crear.add_argument("--modo", default="apertura_siguiente", choices=["cierre", "apertura_siguiente"])
    crear.add_argument("--capital", type=float, default=100_000.0)
    crear.add_argument("--nombre")
    crear.add_argument("--comision", type=float)
    crear.add_argument("--slippage", type=float)
    crear.add_argument("--fecha-alta")
    for accion in ("pausar", "reanudar", "detener"):
        ordenes.add_parser(accion).add_argument("sesion")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    servicio = _servicio()
    if args.orden == "avanzar":
        resultados = servicio.simulador.avanzar_varias([args.sesion], args.hasta) if args.sesion \
            else servicio.simulador.avanzar_todas(args.hasta)
        for r in resultados:
            estado = f"ERROR {r.error}" if r.error else (f"omitida ({r.omitida})" if r.omitida else
                                                         f"{r.dias_procesados} día(s) hasta {r.hasta}" if r.dias_procesados else "al día")
            print(f"{r.id_sesion}: {estado}")
            for aviso in r.avisos:
                print(f"    aviso: {aviso}")
        if not resultados:
            print("No hay sesiones activas.")
        return 1 if any(r.error for r in resultados) else 0
    if args.orden == "estado":
        for fila in servicio.listar():
            retorno = f"{fila['retorno_total']:+.2%}" if fila["retorno_total"] is not None else "-"
            print(f"{fila['id']}: {fila['estado']}, {fila['dias']} días, retorno {retorno}, "
                  f"{'al día' if fila['al_dia'] else str(fila['dias_pendientes']) + ' días pendientes'}")
        return 0
    if args.orden == "crear":
        almacen_experimentos = AlmacenExperimentos(RaizDatos().experimentos)
        sesion = servicio.crear_sesion(
            receta_de_experimento(almacen_experimentos, args.experimento), nombre=args.nombre, fecha_alta=args.fecha_alta,
            capital_inicial=args.capital, modo_ejecucion=args.modo, comision=args.comision, slippage=args.slippage,
            id_experimento=args.experimento)
        print(f"Sesión creada: {sesion.id} (alta {sesion.fecha_alta}, modo {sesion.modo_ejecucion})")
        return 0
    sesion = servicio.cambiar_estado(args.sesion, args.orden)
    print(f"{sesion.id}: {sesion.estado}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
