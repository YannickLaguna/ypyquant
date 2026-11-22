"""
calendario.py — Calendario de rebalanceo único (motor de backtest, bot de Alpaca y paper interno).

Responsabilidad:
    Única fuente de verdad de CUÁNDO se rebalancea. Define las frecuencias válidas
    (`FRECUENCIAS_VALIDAS`) y las dos preguntas que antes se contestaban en sitios distintos:
    - «¿qué fechas de este rango son de rebalanceo?» (`fechas_de_rebalanceo`): lo usa el motor de
      backtest, que conoce de antemano todos los días de trading;
    - «¿hoy toca rebalancear, dado el último rebalanceo?» (`toca_rebalancear`): lo usa el bot, que
      decide día a día y recuerda solo la fecha del último rebalanceo.
    Ambas parten de la MISMA definición (`fechas_programadas`: inicio de semana lunes / de mes / de
    trimestre, o cada día), así que no pueden divergir: una fecha programada se ejecuta el primer día
    de trading >= ella (convención de bt `RunMonthly` y zipline `date_rules.month_start`).

    Paridad (test en tests/test_calendario.py): con los mismos días hábiles, la secuencia «toca» del
    bot día a día coincide con las fechas del motor, también con festivos, semanas ISO que cruzan el
    año y rangos que empiezan a mitad de periodo. Solo difieren en una cosa que es una decisión
    deliberada: tras un parón largo (PC apagado) el bot hace UN rebalanceo de recuperación, no uno
    por cada periodo perdido (el motor nunca se para, así que no tiene ese caso).

    Módulo puro (solo pandas y fechas): sin red, sin estado y sin importar nada del resto de
    ypyquant, para que lo usen el motor (`portfolio`) y el bot (`pipelines`) sin dependencias circulares.

Se relaciona con:
    - ypyquant.portfolio.backtest_engine: `_get_rebalance_dates`, `_alinear_a_dias_de_trading` y la
      validación de `BacktestConfig` delegan aquí (su interfaz pública no cambia).
    - ypyquant.estrategias.base: reexporta `FRECUENCIAS_VALIDAS`.
    - pipelines/paper_trading (estado_bot.py y alpaca_vivo.py, CLI `--frecuencia`) y el paper interno
      del plan docs/planes/2026-11-20-configuraciones-y-paper-interno.md (fase 3).
"""

from collections.abc import Sequence
from datetime import date, datetime, timedelta

import pandas as pd

# Alias de pandas de cada frecuencia: cada día / lunes de cada semana ISO / día 1 de cada mes /
# día 1 de enero, abril, julio y octubre. Es el único sitio donde se enumeran las frecuencias.
ALIAS_PANDAS = {"daily": "D", "weekly": "W-MON", "monthly": "MS", "quarterly": "QS"}

FRECUENCIAS_VALIDAS = tuple(ALIAS_PANDAS)


def validar_frecuencia(frecuencia: str) -> str:
    """Devuelve la frecuencia si es válida; si no, ValueError con las admitidas."""
    if frecuencia not in ALIAS_PANDAS:
        raise ValueError(f"Frecuencia no soportada: {frecuencia!r} (frecuencias válidas: {', '.join(FRECUENCIAS_VALIDAS)})")
    return frecuencia


def fechas_programadas(inicio, fin, frecuencia: str) -> list[datetime]:
    """
    Fechas de calendario (no necesariamente hábiles) de rebalanceo entre `inicio` y `fin` inclusive:
    cada día, cada lunes, cada día 1 de mes o cada inicio de trimestre natural.
    """
    validar_frecuencia(frecuencia)
    return [d.to_pydatetime() for d in pd.date_range(inicio, fin, freq=ALIAS_PANDAS[frecuencia])]


def alinear_a_dias_de_trading(programadas: Sequence, dias_trading: Sequence) -> list[pd.Timestamp]:
    """
    Lleva cada fecha programada al primer día de trading >= ella y quita duplicados (dos fechas de
    calendario que caen en el mismo día hábil, p. ej. sábado y domingo con frecuencia diaria). Las
    posteriores al último día de trading se descartan.
    """
    if not len(dias_trading):
        return []
    dias = pd.DatetimeIndex(dias_trading)
    posiciones = dias.searchsorted(pd.DatetimeIndex(programadas), side="left")
    return sorted({dias[i] for i in posiciones if i < len(dias)})


def fechas_de_rebalanceo(inicio, fin, frecuencia: str, dias_trading: Sequence) -> list[pd.Timestamp]:
    """
    Días de trading en que se rebalancea en el rango: las fechas programadas alineadas al primer día
    hábil. Incluye el primer día de trading si coincide con una fecha programada (el motor no lo
    cuenta como rebalanceo porque ese día hace la asignación inicial).
    """
    return alinear_a_dias_de_trading(fechas_programadas(inicio, fin, frecuencia), dias_trading)


def periodo_de_rebalanceo(frecuencia: str, fecha: date) -> tuple[int, ...]:
    """
    Periodo de calendario al que pertenece `fecha`: dos fechas del mismo periodo comparten
    rebalanceo. daily = el día; weekly = semana ISO; monthly = mes; quarterly = trimestre.
    Etiqueta informativa; la decisión de rebalancear sale de `fechas_programadas`.
    """
    validar_frecuencia(frecuencia)
    if frecuencia == "daily":
        return (fecha.year, fecha.month, fecha.day)
    if frecuencia == "weekly":
        iso = fecha.isocalendar()
        return (iso.year, iso.week)
    if frecuencia == "monthly":
        return (fecha.year, fecha.month)
    return (fecha.year, (fecha.month - 1) // 3 + 1)


def toca_rebalancear(frecuencia: str, ultimo: date | None, hoy: date) -> bool:
    """
    True si hoy toca rebalancear: nunca se rebalanceó, o desde el último rebalanceo (excluido) hasta
    hoy (incluido) hubo alguna fecha programada. Llamada cada día hábil, dispara exactamente el primer
    día de trading de cada periodo, como el motor; tras un parón largo dispara una sola vez (recupera
    el rebalanceo perdido). Si el reloj va por detrás de la última fecha registrada, no rebalancea.
    """
    validar_frecuencia(frecuencia)
    if ultimo is None:
        return True
    if hoy <= ultimo:
        return False
    return bool(fechas_programadas(ultimo + timedelta(days=1), hoy, frecuencia))
