"""
Receta de un experimento: el entorno y la estrategia, con UNA sola canonización y UNA sola huella.

Responsabilidad:
- `Entorno` (Pydantic): lo que se mantiene fijo para comparar estrategias entre sí (ceteris paribus): universo,
  frecuencia de rebalanceo, costes (comisión, slippage), benchmark, semilla y días de historia previa.
  `normalizar_universo` es la única regla de limpieza del universo en lista (mayúsculas, sin duplicados, no vacío).
- `Receta` (Pydantic) = `Entorno` + estrategia (nombre + parámetros). Es lo que una configuración guardada congela y lo
  que un marco compone al aplicarse a una estrategia. `normalizada()` valida los parámetros y completa los defectos y
  la frecuencia; `huella()` es el SHA-256 de su JSON canónico.
- Canonización única (`huella_de`, `valor_comparable`): la usan la huella de la receta, la del experimento completo
  (`ConfiguracionExperimento` = receta + periodo + capital + validaciones, ver `experimentos.py`) y la comparación
  ceteris paribus de `marcos.py`. Reglas: claves ordenadas, universo en lista sin orden, y los parámetros que una
  estrategia añadió DESPUÉS de los experimentos guardados no entran en la huella mientras valgan su defecto
  (`parametros_fuera_de_huella_si_defecto`).
- La huella de los experimentos ya guardados NO debe cambiar: lo vigila `tests/test_huellas_guardadas.py` contra
  `data/experimentos` y `tests/test_receta.py` con valores fijos.

Se relaciona con:
- ypyquant.estrategias.registro (clase de la estrategia: `Parametros`, frecuencia sugerida).
- ypyquant.estrategias.experimentos (`ConfiguracionExperimento` hereda de `Receta`), .configuraciones
  (`ConfiguracionGuardada` hereda de `Receta`) y .marcos (`MarcoComparacion` hereda de `Entorno`).
- ypyquant.estrategias.servicio_experimentos y el índice por receta de .almacen (agrupan backtests por `huella()`).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ypyquant.estrategias.registro import obtener_estrategia

Frecuencia = Literal["daily", "weekly", "monthly", "quarterly"]

# Campos del entorno (comunes a todas las estrategias de una comparación) y de la estrategia
CAMPOS_ENTORNO = ("universo", "frecuencia_rebalanceo", "comision", "slippage", "benchmark", "semilla",
                  "dias_historia_previa")
CAMPOS_ESTRATEGIA = ("estrategia", "parametros")


def normalizar_universo(universo: list[str] | str) -> list[str] | str:
    """Universo en lista: mayúsculas, sin duplicados (orden de aparición) y no vacío. Un nombre se deja tal cual."""
    if isinstance(universo, list):
        if not universo:
            raise ValueError("el universo no puede estar vacío")
        return list(dict.fromkeys(t.strip().upper() for t in universo if t.strip()))
    return universo


def json_canonico(datos: Any) -> str:
    """JSON determinista: claves ordenadas, sin espacios, UTF-8 sin escapar."""
    return json.dumps(datos, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def valor_comparable(valor: Any) -> str:
    """Forma comparable de un valor de configuración: una lista (el universo) se compara sin orden."""
    return json_canonico(sorted(valor) if isinstance(valor, list) else valor)


def _omitir_parametros_nuevos_en_defecto(datos: dict[str, Any]) -> None:
    """
    Quita de `datos['parametros']` los parámetros que la estrategia declara como posteriores a los experimentos
    guardados (`parametros_fuera_de_huella_si_defecto`) mientras valgan su defecto. Así `normalizada()` puede
    rellenar los parámetros nuevos sin cambiar la huella de ningún experimento anterior.
    """
    try:
        clase = obtener_estrategia(datos["estrategia"])
    except KeyError:  # estrategia no registrada (p. ej. importada del TFM): la huella usa los parámetros tal cual
        return
    for nombre in clase.parametros_fuera_de_huella_si_defecto:
        campo = clase.Parametros.model_fields.get(nombre)
        if campo is not None and nombre in datos["parametros"] and datos["parametros"][nombre] == campo.default:
            del datos["parametros"][nombre]


def huella_de(datos: dict[str, Any]) -> str:
    """
    SHA-256 de la canonización de `datos` (un `model_dump` de receta o de experimento, sin la etiqueta).
    No modifica `datos`. Es la ÚNICA función que calcula huellas: cambiarla invalida los ids de lo guardado.
    """
    copia = {**datos, "parametros": dict(datos.get("parametros") or {})}
    _omitir_parametros_nuevos_en_defecto(copia)
    if isinstance(copia["universo"], list):
        copia["universo"] = sorted(copia["universo"])
    return hashlib.sha256(json_canonico(copia).encode("utf-8")).hexdigest()


class Entorno(BaseModel):
    """
    Condiciones comunes de un backtest, independientes de la estrategia (ver cabecera del módulo).

    `universo`: lista de tickers o nombre (`'sp500'`, `'sp500:30'`, `'sp500_pit'`). `frecuencia_rebalanceo=None`
    deja a cada estrategia con la suya. `comision` / `slippage`: fracciones por operación (0.001 = 10 pb).
    `dias_historia_previa`: días naturales anteriores al inicio que se cargan para estimar.
    """

    model_config = ConfigDict(extra="forbid")

    universo: list[str] | str
    frecuencia_rebalanceo: Frecuencia | None = None
    comision: float = Field(0.001, ge=0, le=0.1)
    slippage: float = Field(0.001, ge=0, le=0.1)
    benchmark: str = "SPY"
    semilla: int = 42
    dias_historia_previa: int = Field(1100, ge=0, le=7300)

    @model_validator(mode="after")
    def _universo_limpio(self) -> Self:
        self.universo = normalizar_universo(self.universo)
        return self

    def entorno(self) -> dict[str, Any]:
        """Los campos del entorno como dict (para componerlos en otro modelo sin copiarlos uno a uno)."""
        return {campo: getattr(self, campo) for campo in CAMPOS_ENTORNO}


class Receta(Entorno):
    """
    Entorno + estrategia: todo lo que define un backtest salvo el periodo y el capital.

    `parametros`: valores de `Estrategia.Parametros` (los que falten usan el defecto al normalizar).
    """

    estrategia: str
    parametros: dict[str, Any] = Field(default_factory=dict)

    def receta(self) -> Receta:
        """La receta pura de este modelo (si es una subclase con más campos, solo entorno y estrategia)."""
        return Receta(**self.model_dump(include=set(CAMPOS_ENTORNO) | set(CAMPOS_ESTRATEGIA)))

    def normalizada(self) -> Self:
        """Copia con los parámetros validados y completados con los defectos, y la frecuencia resuelta."""
        clase = obtener_estrategia(self.estrategia)
        parametros = clase.Parametros(**self.parametros).model_dump()
        return self.model_copy(update={
            "parametros": parametros,
            "frecuencia_rebalanceo": self.frecuencia_rebalanceo or clase.frecuencia_rebalanceo,
        })

    def huella(self) -> str:
        """Hash SHA-256 estable de la receta (el orden del universo no importa)."""
        return huella_de(self.model_dump(include=set(CAMPOS_ENTORNO) | set(CAMPOS_ESTRATEGIA)))
