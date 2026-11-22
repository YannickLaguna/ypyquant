"""
Vínculo cuenta-experimento: qué estrategia opera cada cuenta de Alpaca, con su receta CONGELADA.

Responsabilidad:
- Una cuenta (perfil) opera UN experimento. Al vincularla se congela su `Receta` (estrategia + parámetros + entorno,
  ver ypyquant.estrategias.receta), con la huella, el universo ya resuelto a tickers, la frecuencia de rebalanceo
  (la de la receta o la sugerida por la estrategia), la fecha de alta y el equity inicial de la cuenta. Si el
  experimento se borra o se edita después, el bot sigue operando lo que se congeló.
- `receta_de_origen`: obtiene la receta de un experimento (o, como extra barato, de una configuración guardada).
- `resolver_tickers`: el universo de la receta como lista. El universo dinámico `sp500_pit` NO se soporta en vivo:
  decide la elegibilidad por fecha con la historia del índice, y el bot no la tiene; lanza `UniversoNoSoportado`.
- `validar_operable`: además, el margen: las recetas apalancadas o con cortos exigen una cuenta de margen (`CapacidadMargen`).
- `vincular`: escribe `vinculo.json`. Es idempotente: si la cuenta ya está vinculada a la misma receta y origen, no
  cambia nada (conserva fecha de alta y equity inicial). Si es otra receta, guarda la anterior en
  `vinculos_anteriores`, escribe el vínculo nuevo y borra el estado de la estrategia (es memoria de la anterior).

Se relaciona con:
- ypyquant.estrategias (Receta, ConfiguracionExperimento, AlmacenExperimentos, AlmacenConfiguraciones) y
  ypyquant.data.universos (resolver_universo, es_universo_pit).
- ypyquant.perfiles_alpaca.almacen (donde se guarda) y pipelines/paper_trading/__main__.py (el bot lo crea al arrancar
  con `--experimento`); app/api/rutas_alpaca.py (valida el experimento antes de lanzar el bot y lo muestra en la UI).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ypyquant.data.universos import es_universo_pit, resolver_universo
from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.estrategias.receta import Receta
from ypyquant.execution.margen import SIN_MARGEN, CapacidadMargen, incumplimientos, reglas_desde_parametros
from ypyquant.perfiles_alpaca.almacen import AlmacenPerfiles

TIPOS_ORIGEN = ("experimento", "configuracion")


class RecetaNoOperable(ValueError):
    """La receta no se puede operar en una cuenta de Alpaca (universo dinámico, o margen/cortos que la cuenta no admite)."""


class UniversoNoSoportado(RecetaNoOperable):
    """El universo de la receta no se puede operar en vivo (p. ej. `sp500_pit`)."""


def validar_operable(receta: Receta, capacidad: CapacidadMargen = SIN_MARGEN) -> list[str]:
    """
    Comprueba que la receta se puede operar y devuelve sus tickers. `RecetaNoOperable` si no.

    Las recetas apalancadas (`apalancamiento_maximo` > 1, CASH negativo = préstamo) o con cortos (`exposicion_corta` > 0)
    solo se admiten si la cuenta es de margen (`capacidad`, ver `ypyquant.perfiles_alpaca.perfiles.capacidad_declarada`):
    el apalancamiento no puede pasar del multiplicador de la cuenta (2x overnight como máximo) y los cortos exigen
    ventas en corto habilitadas. Con la capacidad por defecto (sin margen) se rechazan con un mensaje claro.
    """
    problemas = incumplimientos(reglas_desde_parametros(receta.parametros), capacidad)
    if problemas:
        raise RecetaNoOperable("; ".join(problemas))
    return resolver_tickers(receta)


def resolver_tickers(receta: Receta) -> list[str]:
    """Universo de la receta como lista de tickers (mayúsculas, sin duplicados). `UniversoNoSoportado` si es dinámico."""
    if isinstance(receta.universo, str) and es_universo_pit(receta.universo):
        raise UniversoNoSoportado(
            f"El universo dinámico '{receta.universo}' no se puede operar en Alpaca: la elegibilidad por fecha necesita la "
            "historia de membresías del índice. Usa un universo fijo (p. ej. 'sp500:30') o una lista de tickers.")
    tickers = resolver_universo(receta.universo)
    return list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))


def receta_de_origen(tipo: str, id_origen: str, almacen_experimentos: Any,
                     almacen_configuraciones: Any = None, version: int | None = None) -> Receta:
    """
    Receta normalizada (parámetros completos y frecuencia resuelta) de un experimento o de una configuración guardada.

    `KeyError` si no existe; `ValueError`/`ValidationError` si la estrategia o sus parámetros ya no son válidos.
    """
    if tipo == "experimento":
        config = almacen_experimentos.leer_config(id_origen)
        if config is None:
            raise KeyError(f"El experimento '{id_origen}' no existe")
        return ConfiguracionExperimento(**config).receta().normalizada()
    if tipo == "configuracion":
        if almacen_configuraciones is None:
            raise KeyError("No hay almacén de configuraciones")
        try:
            return almacen_configuraciones.leer(id_origen, version).receta().normalizada()
        except (FileNotFoundError, KeyError) as error:
            raise KeyError(f"La configuración '{id_origen}' no existe") from error
    raise ValueError(f"Origen de vínculo desconocido: {tipo!r} (válidos: {', '.join(TIPOS_ORIGEN)})")


def vincular(almacen: AlmacenPerfiles, perfil: str, receta: Receta, tipo: str, id_origen: str,
             equity_inicial: float | None, ahora: datetime | None = None,
             capacidad: CapacidadMargen = SIN_MARGEN) -> tuple[dict[str, Any], bool]:
    """
    Vincula el perfil a la receta y devuelve `(vinculo, cambio)`; `cambio` es False si ya estaba vinculado a lo mismo.

    La receta debe venir normalizada (ver `receta_de_origen`). `capacidad` es la de margen de la cuenta (por defecto, sin
    margen: se rechazan las recetas apalancadas o con cortos). El vínculo anterior, si lo había, pasa a
    `vinculos_anteriores` y se borra el estado persistido de la estrategia.
    """
    validar_operable(receta, capacidad)
    huella = receta.huella()
    actual = almacen.leer_vinculo(perfil)
    if actual and actual.get("huella_receta") == huella and actual.get("origen") == {"tipo": tipo, "id": id_origen}:
        return actual, False
    anteriores = list(actual.get("vinculos_anteriores", [])) if actual else []
    if actual:
        anteriores.append({k: actual.get(k) for k in ("origen", "huella_receta", "fecha_alta", "equity_inicial")})
        almacen.borrar_estado_estrategia(perfil)
    vinculo = {
        "perfil": perfil,
        "origen": {"tipo": tipo, "id": id_origen},
        "estrategia": receta.estrategia,
        "receta": receta.model_dump(),
        "huella_receta": huella,
        "tickers": resolver_tickers(receta),
        "frecuencia_rebalanceo": receta.frecuencia_rebalanceo,
        "fecha_alta": (ahora or datetime.now(UTC)).isoformat(),
        "equity_inicial": equity_inicial,
        "vinculos_anteriores": anteriores,
    }
    almacen.guardar_vinculo(perfil, vinculo)
    return vinculo, True
