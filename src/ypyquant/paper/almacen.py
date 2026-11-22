"""
almacen.py — Persistencia de las sesiones de paper interno y bloqueo entre procesos.

Responsabilidad:
    - `AlmacenPaper`: adaptador fino sobre `AlmacenDocumentos` (`ypyquant.almacen_documentos`): una carpeta por
      sesión en `data/paper_interno/<id>/` con `sesion.json` (la marca que define que la carpeta ES una sesión),
      `estado.json` (el `EstadoCartera`; se escribe el ÚLTIMO en cada día, es el commit), `eventos.json` (avisos y
      sucesos: re-bases, liquidaciones, errores de estrategia) y las tablas Parquet `equity`, `operaciones`,
      `pesos` y `rotaciones` (`TablasSesion`). El formato de las tablas es el de un experimento (más columnas
      extra), para que `simulacion_desde_tablas` + `evaluar` las traten igual que un backtest.
    - Reanudación idempotente: al leer las tablas se descartan las filas posteriores a `ultima_fecha_procesada`
      (restos de un día que cayó antes de escribir `estado.json`); el día se repite entero sin duplicar filas.
    - `bloqueo(id)`: cerrojo entre PROCESOS (programador de Windows, API y CLI manual) con un archivo
      `sesion.lock` creado de forma exclusiva y el pid del dueño. Un bloqueo cuyo proceso ya no existe se roba;
      uno de un proceso vivo (también el propio: otro hilo) lanza `SesionBloqueada`.

Se relaciona con:
    - ypyquant.almacen_documentos (E/S atómica) y ypyquant.raiz_datos (`RaizDatos.paper_interno`).
    - ypyquant.paper.sesion (modelos), ypyquant.paper.simulador (lo usa para avanzar) y .servicio (para consultar).
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ypyquant.almacen_documentos import AlmacenDocumentos, id_slug
from ypyquant.paper.sesion import EstadoCartera, SesionPaper

COLUMNAS_EQUITY = ["fecha", "equity", "efectivo", "n_posiciones", "benchmark"]
COLUMNAS_OPERACIONES = ["fecha", "ticker", "lado", "cantidad", "precio", "comision", "pnl", "slippage"]
COLUMNAS_PESOS = ["fecha", "ticker", "peso"]
COLUMNAS_ROTACIONES = ["fecha", "rotacion"]
_COLUMNAS = {"equity": COLUMNAS_EQUITY, "operaciones": COLUMNAS_OPERACIONES, "pesos": COLUMNAS_PESOS,
             "rotaciones": COLUMNAS_ROTACIONES}


_TIPOS = {"fecha": "datetime64[ns]", "ticker": "object", "lado": "object", "n_posiciones": "int64"}


class SesionBloqueada(RuntimeError):
    """Otro proceso (o hilo) está avanzando esta sesión."""


class SesionInexistente(KeyError):
    """No hay ninguna sesión con ese id."""


@dataclass
class TablasSesion:
    """Las cuatro tablas de una sesión y su lista de eventos, en memoria (el simulador las amplía día a día)."""

    equity: pd.DataFrame
    operaciones: pd.DataFrame
    pesos: pd.DataFrame
    rotaciones: pd.DataFrame
    eventos: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def vacias(cls) -> TablasSesion:
        return cls(*(pd.DataFrame({c: pd.Series(dtype=_TIPOS.get(c, "float64")) for c in _COLUMNAS[n]})
                     for n in ("equity", "operaciones", "pesos", "rotaciones")))

    def recortar(self, hasta: pd.Timestamp) -> bool:
        """Quita las filas con fecha posterior a `hasta`; True si quitó alguna."""
        quitadas = False
        for nombre in _COLUMNAS:
            tabla: pd.DataFrame = getattr(self, nombre)
            if len(tabla) and (tabla["fecha"] > hasta).any():
                setattr(self, nombre, tabla[tabla["fecha"] <= hasta].reset_index(drop=True))
                quitadas = True
        evento_previos = [e for e in self.eventos if pd.Timestamp(e["fecha"]) <= hasta]
        if len(evento_previos) != len(self.eventos):
            self.eventos, quitadas = evento_previos, True
        return quitadas


def _proceso_vivo(pid: int) -> bool:
    """True si el proceso existe. En Windows se usa la API (os.kill(pid, 0) allí TERMINA el proceso)."""
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        manejador = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)  # type: ignore[attr-defined]
        if not manejador:
            return False
        try:
            codigo = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(manejador, ctypes.byref(codigo))  # type: ignore[attr-defined]
            return bool(ok) and codigo.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(manejador)  # type: ignore[attr-defined]
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class AlmacenPaper:
    """Sesiones de paper interno en disco (ver cabecera del módulo)."""

    def __init__(self, raiz: Path | str):
        self.documentos = AlmacenDocumentos(raiz, nombre="sesión de paper", validar_id=id_slug, marca="sesion")
        self.raiz = self.documentos.raiz

    # ---- ciclo de vida -------------------------------------------------------------------------

    def existe(self, id_sesion: str) -> bool:
        return self.documentos.existe(id_sesion)

    def ids(self) -> list[str]:
        return self.documentos.ids()

    def crear(self, sesion: SesionPaper, estado: EstadoCartera) -> None:
        """Crea la carpeta de la sesión con su estado inicial; `FileExistsError` si el id ya existe."""
        with self.documentos.cerrojo:
            if self.documentos.hay_carpeta(sesion.id):
                raise FileExistsError(f"Ya existe la sesión {sesion.id!r}")
            self.documentos.crear(sesion.id)
            self.guardar_tablas(sesion.id, TablasSesion.vacias())
            self.guardar_estado(sesion.id, estado)
            self.guardar_sesion(sesion)  # al final: la marca hace que la sesión «exista»

    def borrar(self, id_sesion: str) -> bool:
        return self.documentos.borrar(id_sesion)

    # ---- sesión y estado -----------------------------------------------------------------------

    def leer_sesion(self, id_sesion: str) -> SesionPaper:
        datos = self.documentos.leer_json(id_sesion, "sesion")
        if datos is None:
            raise SesionInexistente(id_sesion)
        return SesionPaper(**datos)

    def guardar_sesion(self, sesion: SesionPaper) -> None:
        self.documentos.escribir_json(sesion.id, "sesion", sesion.model_dump())

    def sesiones(self) -> list[SesionPaper]:
        return [self.leer_sesion(i) for i in self.ids()]

    def leer_estado(self, id_sesion: str) -> EstadoCartera:
        datos = self.documentos.leer_json(id_sesion, "estado", reintentar_si_falta=True)
        if datos is None:
            raise SesionInexistente(id_sesion)
        return EstadoCartera(**datos)

    def guardar_estado(self, id_sesion: str, estado: EstadoCartera) -> None:
        self.documentos.escribir_json(id_sesion, "estado", estado.model_dump())

    # ---- tablas --------------------------------------------------------------------------------

    def leer_tablas(self, id_sesion: str, hasta: str | pd.Timestamp | None = None) -> TablasSesion:
        """
        Tablas de la sesión. Con `hasta` (la `ultima_fecha_procesada`) descarta las filas posteriores: son de un día
        que no llegó a confirmarse (caída entre escribir las tablas y `estado.json`).
        """
        vacias = TablasSesion.vacias()
        tablas = {}
        for nombre in _COLUMNAS:
            tabla = self.documentos.leer_parquet(id_sesion, nombre)
            tablas[nombre] = tabla if tabla is not None else getattr(vacias, nombre)
        eventos = self.documentos.leer_json(id_sesion, "eventos") or []
        resultado = TablasSesion(eventos=eventos, **tablas)
        if hasta is not None:
            resultado.recortar(pd.Timestamp(hasta))
        return resultado

    def guardar_tablas(self, id_sesion: str, tablas: TablasSesion) -> None:
        for nombre in _COLUMNAS:
            self.documentos.escribir_parquet(id_sesion, nombre, getattr(tablas, nombre))
        self.documentos.escribir_json(id_sesion, "eventos", tablas.eventos)

    # ---- bloqueo entre procesos ----------------------------------------------------------------

    @contextmanager
    def bloqueo(self, id_sesion: str) -> Iterator[None]:
        """Exclusión mutua sobre la sesión; ver cabecera. `SesionBloqueada` si la tiene un proceso vivo."""
        carpeta = self.documentos.carpeta(id_sesion)
        ruta = carpeta / "sesion.lock"
        for _ in range(2):
            try:
                descriptor = os.open(ruta, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(ruta.read_text(encoding="utf-8").strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid and _proceso_vivo(pid):
                    raise SesionBloqueada(f"La sesión {id_sesion} la está avanzando el proceso {pid}") from None
                try:  # dueño muerto (o archivo corrupto): se roba el bloqueo y se reintenta
                    ruta.unlink()
                except FileNotFoundError:
                    pass
                continue
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
            break
        else:
            raise SesionBloqueada(f"No se pudo bloquear la sesión {id_sesion}")
        try:
            yield
        finally:
            try:
                ruta.unlink()
            except FileNotFoundError:
                pass
