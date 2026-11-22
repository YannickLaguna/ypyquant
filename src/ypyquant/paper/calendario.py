"""
calendario.py — Qué sesiones están cerradas y cuáles faltan por procesar en el paper interno.

Responsabilidad:
    Las dos preguntas de tiempo real que el simulador necesita y que no son de rebalanceo (eso es de
    `ypyquant.execution.calendario`, que se reutiliza tal cual):
    - `ultima_sesion_cerrada(ahora)`: último día cuya barra diaria ya es definitiva. Hoy cuenta solo si en
      Nueva York son más de las 16:30 (cierre a las 16:00 + margen); si no, es el día anterior. Un reloj
      inyectable (`ahora`) permite probarlo sin esperar al mercado.
    - `dias_pendientes(ultima_fecha, dias_con_datos, ahora)`: días de trading con datos posteriores a la última
      fecha procesada y ya cerrados (catch-up exacto tras un parón).
    - `dias_pendientes_estimados(ultima_fecha, ahora)`: lo mismo sin datos, contando días laborables (los festivos
      cuentan como pendientes hasta el primer avance); sirve para la etiqueta «n días pendientes» de la lista.
    Módulo puro: sin red ni disco. Los días de trading salen de los datos (unión de fechas con barra, como el
    motor), no de un calendario bursátil externo.

Se relaciona con:
    - ypyquant.paper.simulador (decide qué días procesar) y ypyquant.paper.servicio (pendientes de la lista).
    - ypyquant.execution.calendario (`toca_rebalancear`, el calendario de rebalanceo único).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ZONA_NUEVA_YORK = ZoneInfo("America/New_York")
HORA_DATOS_DEFINITIVOS = time(16, 30)  # cierre 16:00 NY + margen para que el proveedor consolide la barra


def ahora_nueva_york() -> datetime:
    """Reloj por defecto: la hora actual en Nueva York (con zona)."""
    return datetime.now(ZONA_NUEVA_YORK)


def ultima_sesion_cerrada(ahora: datetime) -> date:
    """Último día con barra definitiva: hoy si en Nueva York son más de las 16:30; si no, ayer."""
    ahora_ny = ahora.astimezone(ZONA_NUEVA_YORK) if ahora.tzinfo else ahora.replace(tzinfo=ZONA_NUEVA_YORK)
    return ahora_ny.date() if ahora_ny.time() >= HORA_DATOS_DEFINITIVOS else ahora_ny.date() - timedelta(days=1)


def dias_pendientes(ultima_fecha: date | None, dias_con_datos: Iterable, ahora: datetime) -> list[pd.Timestamp]:
    """Días con datos posteriores a `ultima_fecha` (excluida) y no posteriores a la última sesión cerrada, en orden."""
    tope = pd.Timestamp(ultima_sesion_cerrada(ahora))
    inicio = pd.Timestamp(ultima_fecha) if ultima_fecha else None
    return sorted({pd.Timestamp(d) for d in dias_con_datos
                   if (inicio is None or pd.Timestamp(d) > inicio) and pd.Timestamp(d) <= tope})


def dias_pendientes_estimados(ultima_fecha: date, ahora: datetime) -> int:
    """Días laborables entre `ultima_fecha` (excluida) y la última sesión cerrada (incluida); nunca negativo."""
    tope = ultima_sesion_cerrada(ahora)
    if tope <= ultima_fecha:
        return 0
    return int(np.busday_count(ultima_fecha + timedelta(days=1), tope + timedelta(days=1)))
