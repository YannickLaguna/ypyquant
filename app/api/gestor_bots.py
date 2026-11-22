"""
Gestor de bots de Alpaca: un subproceso por perfil (cuenta paper), con rehidratación al arrancar la API.

Responsabilidad:
- `GestorBots`: arranca y para `python -m pipelines.paper_trading --perfil X ...` (un bot por perfil). Cada bot guarda su
  salida en `data/paper_trading/<PERFIL>/bot.log` (el log SE REDACTA al servirlo: ninguna clave sale por la API) y deja
  `bot.json` con el pid, la marca (hora de creación) del proceso, el modo, el intervalo y el experimento.
- Un perfil ocupado devuelve `PerfilOcupado` (409). Más de `limite_activos` (3) bots a la vez devuelve `LimiteBotsActivos`
  (límite blando: protege de rate limits y de olvidarse bots abiertos; se sube al construir el gestor).
- `rehidratar()` (la llama la API al arrancar): lee los `bot.json`, comprueba que el pid siga vivo y sea el mismo proceso
  (pid + hora de creación) y lo adopta; descarta los muertos. Así, si la API se reinicia, el bot huérfano no se pierde
  de vista (en Windows el hijo sobrevive a la API). Parar un bot adoptado se hace por pid.
- Modos del bot: `continuo` (bucle cada `intervalo_s`) y `diario` (una pasada y sale; para el programador de tareas).

Se relaciona con:
- app/api/rutas_alpaca.py (endpoints), app/api/contexto.py (lo cuelga del contexto) y app/api/paper.py (`BotControl`, el
  bot único heredado, que sigue en `/api/paper/bot/*`).
- ypyquant.perfiles_alpaca (almacén de perfiles, procesos, redacción y límite) y pipelines/paper_trading (el CLI).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ypyquant.perfiles_alpaca.almacen import AlmacenPerfiles
from ypyquant.perfiles_alpaca.perfiles import LIMITE_PERFILES_ACTIVOS
from ypyquant.perfiles_alpaca.procesos import marca_de_proceso, proceso_vivo, terminar
from ypyquant.perfiles_alpaca.redaccion import redactar

RAIZ_REPO = Path(__file__).resolve().parents[2]
MODOS = ("continuo", "diario")


class PerfilOcupado(RuntimeError):
    """El perfil ya tiene un bot en marcha."""


class LimiteBotsActivos(RuntimeError):
    """Ya hay el máximo de bots simultáneos (límite blando)."""


def comando_por_defecto(perfil: str, experimento_id: str | None, configuracion_id: str | None,
                        modo: str, intervalo_s: int) -> list[str]:
    """Línea de comandos del bot de un perfil; el vínculo (experimento o configuración) siempre se pasa explícito."""
    comando = [sys.executable, "-m", "pipelines.paper_trading", "--perfil", perfil, "--modo", modo,
               "--intervalo", str(intervalo_s)]
    if experimento_id:
        comando += ["--experimento", experimento_id]
    elif configuracion_id:
        comando += ["--configuracion", configuracion_id]
    return comando


class GestorBots:
    """
    Bots de paper trading, uno por perfil.

    Args:
        almacen: `AlmacenPerfiles` (carpeta de cada perfil: `bot.json`, `bot.log`).
        construir_comando: `(perfil, experimento_id, configuracion_id, modo, intervalo_s) -> argv`; por defecto
            `comando_por_defecto` (se inyecta otro en los tests).
        limite_activos: máximo de bots a la vez (límite blando).
    """

    def __init__(self, almacen: AlmacenPerfiles, construir_comando: Callable[..., list[str]] | None = None,
                 limite_activos: int = LIMITE_PERFILES_ACTIVOS):
        self.almacen = almacen
        self.construir_comando = construir_comando or comando_por_defecto
        self.limite_activos = limite_activos
        self._procesos: dict[str, subprocess.Popen] = {}
        self._adoptados: dict[str, dict[str, Any]] = {}
        self._codigos: dict[str, int | None] = {}
        self._cerrojo = threading.RLock()

    # ---- consulta --------------------------------------------------------------------------------

    def _pid_vivo_de_meta(self, meta: dict[str, Any]) -> bool:
        return proceso_vivo(meta.get("pid"), meta.get("marca_proceso"))

    def en_marcha(self, perfil: str) -> bool:
        """True si el bot del perfil sigue vivo (hijo de esta API o adoptado tras un reinicio)."""
        with self._cerrojo:
            proceso = self._procesos.get(perfil)
            if proceso is not None:
                if proceso.poll() is None:
                    return True
                self._codigos[perfil] = proceso.returncode
                del self._procesos[perfil]
                self.almacen.borrar_bot(perfil)
                return False
            meta = self._adoptados.get(perfil)
            if meta is not None:
                if self._pid_vivo_de_meta(meta):
                    return True
                del self._adoptados[perfil]
                self.almacen.borrar_bot(perfil)
            return False

    def activos(self) -> list[str]:
        """Perfiles con un bot en marcha."""
        with self._cerrojo:
            return sorted(p for p in {*self._procesos, *self._adoptados} if self.en_marcha(p))

    def estado(self, perfil: str) -> dict[str, Any]:
        """`{perfil, en_marcha, pid, codigo_salida, adoptado, modo, intervalo_s, experimento_id, configuracion_id, inicio}`."""
        with self._cerrojo:
            marcha = self.en_marcha(perfil)
            meta = self.almacen.leer_bot(perfil) if marcha else None
            base = {"perfil": perfil, "en_marcha": marcha, "pid": None, "codigo_salida": self._codigos.get(perfil),
                    "adoptado": perfil in self._adoptados, "modo": None, "intervalo_s": None,
                    "experimento_id": None, "configuracion_id": None, "inicio": None}
            if meta:
                base.update({k: meta.get(k) for k in ("pid", "modo", "intervalo_s", "experimento_id", "configuracion_id", "inicio")})
            return base

    # ---- ciclo de vida ---------------------------------------------------------------------------

    def rehidratar(self) -> list[str]:
        """
        Adopta los bots que siguen vivos según su `bot.json` y limpia los de procesos muertos o reutilizados.

        Devuelve los perfiles adoptados. Se llama una vez al arrancar la API.
        """
        adoptados = []
        with self._cerrojo:
            for perfil in self.almacen.perfiles_con_datos():
                meta = self.almacen.leer_bot(perfil)
                if not meta:
                    continue
                if self._pid_vivo_de_meta(meta):
                    self._adoptados[perfil] = meta
                    adoptados.append(perfil)
                else:
                    self.almacen.borrar_bot(perfil)
        return adoptados

    def arrancar(self, perfil: str, experimento_id: str | None = None, configuracion_id: str | None = None,
                 modo: str = "continuo", intervalo_s: int = 60) -> dict[str, Any]:
        """
        Lanza el bot del perfil. `PerfilOcupado` si ya hay uno; `LimiteBotsActivos` si se supera el límite blando.

        El bot debe llevar un vínculo (experimento o configuración): lo exige el CLI, que lo congela al arrancar.
        """
        if modo not in MODOS:
            raise ValueError(f"Modo desconocido: {modo!r} (válidos: {', '.join(MODOS)})")
        with self._cerrojo:
            if self.en_marcha(perfil):
                raise PerfilOcupado(f"El perfil {perfil} ya tiene un bot en marcha; páralo antes de lanzar otro")
            if len(self.activos()) >= self.limite_activos:
                raise LimiteBotsActivos(
                    f"Ya hay {self.limite_activos} bots en marcha ({', '.join(self.activos())}): es el límite blando de cuentas activas")
            self.almacen.carpeta(perfil)  # la crea
            entorno = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            with self.almacen.ruta_log(perfil).open("wb") as salida:
                proceso = subprocess.Popen(
                    self.construir_comando(perfil, experimento_id, configuracion_id, modo, intervalo_s),
                    cwd=RAIZ_REPO, stdout=salida, stderr=subprocess.STDOUT, env=entorno)
            self._procesos[perfil] = proceso
            self._codigos.pop(perfil, None)
            self.almacen.guardar_bot(perfil, {
                "perfil": perfil, "pid": proceso.pid, "marca_proceso": marca_de_proceso(proceso.pid), "modo": modo,
                "intervalo_s": intervalo_s, "experimento_id": experimento_id, "configuracion_id": configuracion_id,
                "inicio": datetime.now(UTC).isoformat()})
            return self.estado(perfil)

    def parar(self, perfil: str, espera_s: float = 10.0) -> dict[str, Any]:
        """Termina el bot del perfil (hijo o adoptado), esperando `espera_s` antes de matarlo."""
        with self._cerrojo:
            proceso = self._procesos.get(perfil)
            if proceso is not None and proceso.poll() is None:
                proceso.terminate()
                try:
                    proceso.wait(timeout=espera_s)
                except subprocess.TimeoutExpired:
                    proceso.kill()
                    proceso.wait()
            elif perfil in self._adoptados:
                terminar(self._adoptados[perfil]["pid"], espera_s)
            self.en_marcha(perfil)  # limpia el registro y guarda el código de salida
            return self.estado(perfil)

    def parar_todos(self, espera_s: float = 10.0) -> None:
        """Para todos los bots (al cerrar la API en tests; en producción los bots siguen vivos a propósito)."""
        for perfil in self.activos():
            self.parar(perfil, espera_s)

    # ---- log -------------------------------------------------------------------------------------

    def log(self, perfil: str, desde: int = 0, maximo: int = 500) -> dict[str, Any]:
        """Líneas del log del perfil desde la línea `desde` (offset para polling), REDACTADAS, y el siguiente offset."""
        ruta = self.almacen.ruta_log(perfil)
        if not ruta.exists():
            return {"lineas": [], "siguiente": 0}
        lineas = ruta.read_text(encoding="utf-8", errors="replace").splitlines()
        trozo = [redactar(linea) for linea in lineas[desde:desde + maximo]]
        return {"lineas": trozo, "siguiente": desde + len(trozo)}
