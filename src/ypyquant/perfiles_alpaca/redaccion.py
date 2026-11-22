"""
Redacción de secretos: que ninguna clave de Alpaca llegue a un log, a la API ni a la pantalla.

Responsabilidad:
- `secretos_del_entorno`: valores de las variables `ALPACA_*_API_KEY` / `ALPACA_*_SECRET_KEY` (y las sin nombre).
- `redactar(texto, secretos)`: sustituye cada secreto por `***` y, además, cualquier cadena con forma de API key de
  Alpaca (`PK…`/`AK…` de 16+ caracteres) aunque no esté en el entorno de este proceso.
- `FiltroRedaccion`: `logging.Filter` que redacta el mensaje ya formateado (con argumentos y excepciones); el bot lo
  instala en sus manejadores (`instalar_en_raiz`). El gestor de bots vuelve a redactar al servir el log.

Se relaciona con:
- pipelines/paper_trading/__main__.py (instala el filtro) y app/api/gestor_bots.py (redacta el log servido).
- ypyquant.perfiles_alpaca.perfiles (qué variables contienen secretos).
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping

MARCA = "***"
_VARIABLE_SECRETA = re.compile(r"ALPACA_(?:[A-Z0-9_]+_)?(?:API|SECRET)_KEY")
_FORMA_API_KEY = re.compile(r"\b(?:PK|AK)[A-Z0-9]{16,}\b")
_LONGITUD_MINIMA = 8  # valores más cortos son marcadores de posición: ocultarlos estropearía el texto


def secretos_del_entorno(entorno: Mapping[str, str] | None = None) -> list[str]:
    """Valores (de más largo a más corto, para no dejar restos) de las variables de claves definidas en el entorno."""
    entorno = os.environ if entorno is None else entorno
    valores = {v.strip() for k, v in entorno.items() if _VARIABLE_SECRETA.fullmatch(k) and len(v.strip()) >= _LONGITUD_MINIMA}
    return sorted(valores, key=len, reverse=True)


def redactar(texto: str, secretos: Iterable[str] | None = None) -> str:
    """Texto sin los `secretos` (por defecto, los del entorno) ni nada con forma de API key de Alpaca."""
    for secreto in (secretos_del_entorno() if secretos is None else secretos):
        if secreto:
            texto = texto.replace(secreto, MARCA)
    return _FORMA_API_KEY.sub(MARCA, texto)


class FiltroRedaccion(logging.Filter):
    """Filtro de logging que oculta los secretos en el mensaje (ya formateado) de cada registro y en su traza."""

    def __init__(self, secretos: Iterable[str] | None = None):
        super().__init__()
        self._secretos = list(secretos) if secretos is not None else None

    def filter(self, record: logging.LogRecord) -> bool:
        secretos = self._secretos if self._secretos is not None else secretos_del_entorno()
        record.msg = redactar(record.getMessage(), secretos)
        record.args = None
        if record.exc_info:  # la traza se formatea después: se fija ya el texto redactado
            record.exc_text = redactar(logging.Formatter().formatException(record.exc_info), secretos)
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redactar(record.exc_text, secretos)
        return True


def instalar_en_raiz(secretos: Iterable[str] | None = None) -> FiltroRedaccion:
    """Añade el filtro a todos los manejadores del logger raíz (llamar tras `logging.basicConfig`)."""
    filtro = FiltroRedaccion(secretos)
    for manejador in logging.getLogger().handlers:
        manejador.addFilter(filtro)
    return filtro
