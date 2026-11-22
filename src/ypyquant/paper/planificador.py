"""
planificador.py — Temporizador en segundo plano que pone al día el paper interno mientras la API está en marcha.

Responsabilidad:
    `PlanificadorPaper(servicio, control, intervalo_s)` comprueba cada `intervalo_s` segundos si alguna sesión ACTIVA
    tiene días cerrados pendientes (día cerrado = regla de las 16:30 de Nueva York de `calendario`) y, si es así, la
    avanza con el `ControlAvance` existente, el mismo que usa el botón «Avanzar ahora» y el catch-up del arranque: si ya
    hay un avance en curso NO lanza otro (no se solapan). No sustituye a una tarea programada del sistema: solo avanza
    mientras la API vive (el catch-up exacto del simulador recupera después los días perdidos).
    - `comprobar()` (una comprobación, síncrona y probable sin hilo): devuelve y guarda la última ejecución
      `{hora, resultado, dias_procesados, error}`. Resultados: `al_dia`, `avanzado`, `avance_en_curso`, `error`,
      `sin_datos` (había pendientes pero no se procesó ningún día, p. ej. festivo o proveedor sin la barra) y
      `sin_novedades` (tras `max_intentos_sin_novedad` intentos seguidos sin datos para la misma sesión cerrada
      objetivo deja de descargar hasta que el objetivo cambia: un festivo no debe descargar cada 30 minutos).
    - `iniciar()` / `detener()`: hilo daemon con espera interrumpible; `detener` es limpio (no deja el hilo vivo).
    - `estado()`: `{activo, intervalo_min, proxima_comprobacion, ultima}` para la web.
    La primera comprobación ocurre pasado un intervalo: el arranque ya hace su propio catch-up.
    - Membresías del S&P 500: si se le da un `ServicioMembresias`, cada comprobación mira además si toca refrescarlas
      (como máximo una vez al día, tras el cierre de NY) y, de tocar, lo hace en un hilo aparte para no bloquear el avance
      del paper; un fallo no afecta al paper. Desactivable con `YPYQUANT_MEMBRESIAS_AUTO=0` (ver `membresias_sp500`).

Se relaciona con:
    - ypyquant.data.membresias_sp500 (`ServicioMembresias`: cuándo toca y la comprobación en sí).
    - ypyquant.paper.servicio (`ServicioPaper.hay_dias_pendientes`, `avanzar`, `ControlAvance`) y .calendario
      (`ultima_sesion_cerrada`).
    - app/api/main.py (lo arranca y lo para en el lifespan; `YPYQUANT_PAPER_INTERVALO_MIN`), app/api/contexto.py y
      app/api/rutas_paper_interno.py (`GET /planificador`); app/web: página «Paper interno».
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from ypyquant.data.membresias_sp500 import ServicioMembresias
from ypyquant.paper.calendario import ultima_sesion_cerrada
from ypyquant.paper.servicio import ControlAvance, ServicioPaper

logger = logging.getLogger(__name__)

MAX_INTENTOS_SIN_NOVEDAD = 4


def _reloj_local() -> datetime:
    return datetime.now().astimezone()


class PlanificadorPaper:
    """Comprueba periódicamente si hay días pendientes y avanza las sesiones activas (ver cabecera del módulo)."""

    def __init__(self, servicio: ServicioPaper, control: ControlAvance, intervalo_s: float,
                 reloj: Callable[[], datetime] = _reloj_local, max_intentos_sin_novedad: int = MAX_INTENTOS_SIN_NOVEDAD,
                 membresias: ServicioMembresias | None = None):
        self.membresias = membresias
        self._hilo_membresias: threading.Thread | None = None
        self.servicio = servicio
        self.control = control
        self.intervalo_s = intervalo_s
        self.reloj = reloj
        self.max_intentos_sin_novedad = max_intentos_sin_novedad
        self._ultima: dict[str, Any] | None = None
        self._proxima: datetime | None = None
        self._intentos: dict[Any, int] = {}
        self._parar = threading.Event()
        self._hilo: threading.Thread | None = None
        self._cerrojo = threading.Lock()

    # ---- una comprobación ----------------------------------------------------------------------

    def comprobar(self) -> dict[str, Any]:
        """Una comprobación: avanza si hay días cerrados pendientes y no hay otro avance en curso. Devuelve la ejecución."""
        ejecucion: dict[str, Any] = {"hora": self.reloj().isoformat(timespec="seconds"), "resultado": "al_dia",
                                     "dias_procesados": 0, "error": None}
        try:
            self._comprobar_en(ejecucion)
        except Exception as fallo:  # noqa: BLE001 - un fallo de una comprobación no debe parar el temporizador
            logger.exception("Falló la comprobación automática del paper interno")
            ejecucion.update(resultado="error", error=f"{type(fallo).__name__}: {fallo}")
        with self._cerrojo:
            self._ultima = ejecucion
        self._lanzar_membresias()
        return ejecucion

    def _lanzar_membresias(self) -> None:
        """Si toca (una vez al día tras el cierre de NY) refresca las membresías en un hilo daemon; nunca propaga errores."""
        try:
            if self.membresias is None or (self._hilo_membresias is not None and self._hilo_membresias.is_alive()):
                return
            if self.membresias.toca_comprobar(self.servicio.ahora()):
                self._hilo_membresias = threading.Thread(target=self.membresias.comprobar, name="membresias-sp500", daemon=True)
                self._hilo_membresias.start()
        except Exception:  # noqa: BLE001 - las membresías no deben romper el avance del paper
            logger.exception("Falló el lanzamiento de la comprobación de membresías")

    def _comprobar_en(self, ejecucion: dict[str, Any]) -> None:
        if not self.servicio.hay_dias_pendientes():
            self._intentos.clear()
            return
        objetivo = ultima_sesion_cerrada(self.servicio.ahora())
        if self._intentos.get(objetivo, 0) >= self.max_intentos_sin_novedad:
            ejecucion["resultado"] = "sin_novedades"
            return
        if not self.control.iniciar():
            ejecucion["resultado"] = "avance_en_curso"
            return
        self.control.ejecutar(self.servicio.avanzar)
        estado = self.control.estado()
        resultados = estado["resultados"]
        ejecucion["dias_procesados"] = sum(r["dias_procesados"] for r in resultados)
        error = estado["error"] or next((r["error"] for r in resultados if r["error"]), None)
        if error:
            ejecucion.update(resultado="error", error=error)
        elif ejecucion["dias_procesados"] == 0:
            self._intentos[objetivo] = self._intentos.get(objetivo, 0) + 1
            ejecucion["resultado"] = "sin_datos"
        else:
            self._intentos.pop(objetivo, None)
            ejecucion["resultado"] = "avanzado"

    # ---- hilo ----------------------------------------------------------------------------------

    def iniciar(self) -> None:
        """Arranca el hilo (idempotente). La primera comprobación es pasado un intervalo."""
        if self._hilo is not None and self._hilo.is_alive():
            return
        self._parar.clear()
        self._hilo = threading.Thread(target=self._bucle, name="planificador-paper", daemon=True)
        self._hilo.start()

    def detener(self, espera_s: float = 10.0) -> None:
        """Para el hilo y espera a que termine (una comprobación en curso acaba antes de salir)."""
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(espera_s)
        self._hilo = None
        with self._cerrojo:
            self._proxima = None

    def _bucle(self) -> None:
        while True:
            with self._cerrojo:
                self._proxima = self.reloj() + timedelta(seconds=self.intervalo_s)
            if self._parar.wait(self.intervalo_s):
                return
            self.comprobar()

    def estado(self) -> dict[str, Any]:
        """Lo que muestra la web: si está activo, cada cuánto, cuándo es la próxima comprobación y la última ejecución."""
        with self._cerrojo:
            activo = self._hilo is not None and self._hilo.is_alive()
            return {"activo": activo, "intervalo_min": self.intervalo_s / 60,
                    "proxima_comprobacion": self._proxima.isoformat(timespec="seconds") if activo and self._proxima else None,
                    "ultima": dict(self._ultima) if self._ultima else None}
