"""
Procesos del sistema: saber si un pid sigue vivo y es el MISMO proceso que lanzamos, y terminarlo.

Responsabilidad:
- `marca_de_proceso(pid)`: hora de creación del proceso (segundos desde la época), o None si no se puede obtener. Se
  guarda en `bot.json` junto al pid: un pid se reutiliza tras morir el proceso, la pareja pid + hora de creación no.
- `proceso_vivo(pid, marca)`: True si el pid existe, no es un zombi y (si hay marca) coincide su hora de creación.
- `terminar(pid)`: pide al proceso que acabe (en Windows, TerminateProcess) y espera un poco.
Usa `psutil` si está instalado (viene con el entorno de la API); si no, ctypes en Windows y `os.kill` en POSIX.

Se relaciona con: app/api/gestor_bots.py (rehidratar bots al arrancar la API y pararlos sin tener el `Popen`).
"""

from __future__ import annotations

import os
import signal
import sys
import time

try:  # psutil es opcional: sin él, solo se comprueba que el pid exista
    import psutil
except ImportError:  # pragma: no cover - depende del entorno
    psutil = None

TOLERANCIA_MARCA_S = 2.0


def marca_de_proceso(pid: int) -> float | None:
    """Hora de creación del proceso (epoch, segundos) o None si no existe o no hay `psutil`."""
    if psutil is None:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _existe_sin_psutil(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        acceso_consulta_limitada, estado_activo = 0x1000, 259
        kernel32 = ctypes.windll.kernel32
        manejador = kernel32.OpenProcess(acceso_consulta_limitada, False, pid)
        if not manejador:
            return False
        try:
            codigo = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(manejador, ctypes.byref(codigo))) and codigo.value == estado_activo
        finally:
            kernel32.CloseHandle(manejador)
    try:
        os.kill(pid, 0)  # señal 0: solo comprueba existencia (en Windows sería CTRL_C_EVENT: por eso la rama de arriba)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def proceso_vivo(pid: int | None, marca: float | None = None) -> bool:
    """True si `pid` está vivo y, cuando se conoce la `marca` y se puede medir, es el mismo proceso que se lanzó."""
    if not pid or pid <= 0:
        return False
    if psutil is None:
        return _existe_sin_psutil(pid)
    try:
        proceso = psutil.Process(pid)
        if proceso.status() == psutil.STATUS_ZOMBIE:
            return False
        return marca is None or abs(proceso.create_time() - marca) <= TOLERANCIA_MARCA_S
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def terminar(pid: int, espera_s: float = 10.0) -> bool:
    """Termina el proceso y espera hasta `espera_s`; lo mata si no responde. True si ya no está vivo."""
    try:
        os.kill(pid, signal.SIGTERM)  # en Windows, TerminateProcess
    except (ProcessLookupError, PermissionError, OSError):
        return not proceso_vivo(pid)
    limite = time.monotonic() + espera_s
    while time.monotonic() < limite:
        if not proceso_vivo(pid):
            return True
        time.sleep(0.1)
    if psutil is not None:
        try:
            psutil.Process(pid).kill()
        except psutil.NoSuchProcess:
            pass
    return not proceso_vivo(pid)
