"""
Perfiles de claves de Alpaca: descubrirlos desde el entorno sin exponer nunca un secreto.

Responsabilidad:
- Un PERFIL es una cuenta paper de Alpaca con nombre. Sus claves viven en el `.env` como
  `ALPACA_<PERFIL>_API_KEY` / `ALPACA_<PERFIL>_SECRET_KEY` (PERFIL en mayúsculas, `[A-Z0-9_]`).
  `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (sin nombre) son el perfil `DEFECTO` (compatibilidad con el bot anterior).
- `ALPACA_PERFILES=A,B,C` (opcional) DECLARA perfiles aunque aún no tengan claves: así la UI puede mostrar
  «sin configurar» con las variables que faltan. No hace falta si las claves ya están puestas.
- `descubrir_perfiles` devuelve `PerfilAlpaca` (nombre, si está configurado, últimos 4 caracteres de la API key y qué
  variables faltan). Es lo ÚNICO que la API puede mostrar: nunca el secreto ni la API key completa.
- `claves_de` es la única función que devuelve las claves completas; la usan el cliente de Alpaca y el bot, y no
  debe usarse para mostrar nada.
- `LIMITE_PERFILES_ACTIVOS` (3): límite blando de bots simultáneos (lo aplica el gestor de bots, no esto).

Se relaciona con:
- app/api/paper.py (`ServicioAlpaca(perfil)`), app/api/gestor_bots.py (límite) y app/api/rutas_alpaca.py (lista los perfiles).
- pipelines/paper_trading/config.py (claves del bot por perfil) y .env.example (nombres de las variables).
- ypyquant.perfiles_alpaca.redaccion (qué valores del entorno hay que ocultar en los logs).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

NOMBRE_DEFECTO = "DEFECTO"
LIMITE_PERFILES_ACTIVOS = 3
_NOMBRE_VALIDO = re.compile(r"[A-Z0-9][A-Z0-9_]*")
_VARIABLE_API = re.compile(r"ALPACA_([A-Z0-9][A-Z0-9_]*)_API_KEY")


def nombre_valido(perfil: str) -> bool:
    """Nombre de perfil válido: mayúsculas, dígitos y `_`, sin empezar por `_`. (Es también el nombre de su carpeta.)"""
    return _NOMBRE_VALIDO.fullmatch(perfil) is not None


def variables_de(perfil: str) -> tuple[str, str]:
    """Nombres de las dos variables de entorno `(api_key, secret_key)` de un perfil (`DEFECTO` usa las sin nombre)."""
    if perfil == NOMBRE_DEFECTO:
        return "ALPACA_API_KEY", "ALPACA_SECRET_KEY"
    return f"ALPACA_{perfil}_API_KEY", f"ALPACA_{perfil}_SECRET_KEY"


@dataclass(frozen=True)
class PerfilAlpaca:
    """
    Estado de un perfil, apto para mostrarse: no contiene ningún secreto.

    `ultimos4` son los 4 últimos caracteres de la API key (nunca del secreto). `faltan` son los nombres de las variables
    de entorno que hay que definir (vacío si está configurado).
    """

    nombre: str
    configurado: bool
    ultimos4: str | None = None
    faltan: tuple[str, ...] = field(default_factory=tuple)

    def a_dict(self) -> dict[str, object]:
        api, secreto = variables_de(self.nombre)
        return {"nombre": self.nombre, "configurado": self.configurado, "ultimos4": self.ultimos4,
                "faltan": list(self.faltan), "variables": [api, secreto]}


def claves_de(perfil: str, entorno: Mapping[str, str] | None = None) -> tuple[str, str] | None:
    """`(api_key, secret_key)` del perfil, o None si falta alguna. SOLO para crear clientes: jamás para mostrarlas."""
    entorno = os.environ if entorno is None else entorno
    api, secreto = variables_de(perfil)
    valor_api, valor_secreto = entorno.get(api, "").strip(), entorno.get(secreto, "").strip()
    return (valor_api, valor_secreto) if valor_api and valor_secreto else None


def _perfil(nombre: str, entorno: Mapping[str, str]) -> PerfilAlpaca:
    api, secreto = variables_de(nombre)
    faltan = tuple(v for v in (api, secreto) if not entorno.get(v, "").strip())
    if faltan:
        return PerfilAlpaca(nombre, False, None, faltan)
    return PerfilAlpaca(nombre, True, entorno[api].strip()[-4:], ())


def descubrir_perfiles(entorno: Mapping[str, str] | None = None) -> list[PerfilAlpaca]:
    """
    Perfiles visibles: los que tienen alguna variable de claves, los declarados en `ALPACA_PERFILES` y, si no hay
    ninguno, `DEFECTO` (sin configurar) para que la UI explique qué definir. Orden: `DEFECTO` primero, luego alfabético.

    Un perfil con solo una de las dos variables aparece como NO configurado y con la que falta en `faltan`
    (las variables incompletas nunca se usan). Nombres inválidos se ignoran.
    """
    entorno = os.environ if entorno is None else entorno
    nombres: set[str] = set()
    if entorno.get("ALPACA_API_KEY") or entorno.get("ALPACA_SECRET_KEY"):
        nombres.add(NOMBRE_DEFECTO)
    for variable in entorno:
        if (coincidencia := _VARIABLE_API.fullmatch(variable)):
            nombres.add(coincidencia.group(1))
        elif variable.startswith("ALPACA_") and variable.endswith("_SECRET_KEY") and variable != "ALPACA_SECRET_KEY":
            nombres.add(variable[len("ALPACA_"):-len("_SECRET_KEY")])
    for declarado in entorno.get("ALPACA_PERFILES", "").replace(";", ",").split(","):
        if declarado.strip():
            nombres.add(declarado.strip().upper())
    nombres = {n for n in nombres if nombre_valido(n)}
    if not nombres:
        nombres = {NOMBRE_DEFECTO}
    ordenados = sorted(nombres, key=lambda n: (n != NOMBRE_DEFECTO, n))
    return [_perfil(n, entorno) for n in ordenados]
