"""
test_calendario.py — Calendario de rebalanceo: casos límite y paridad motor/bot.

Responsabilidad:
    Fijar el calendario de rebalanceo (fin de semana, festivo, cambio de semana ISO / mes / trimestre /
    año) y comprobar la PARIDAD entre sus dos formas de uso: el motor de backtest, que calcula de una
    vez las fechas de un rango, y el bot, que decide día a día si hoy toca dado el último rebalanceo.
    Para el mismo conjunto de días hábiles la secuencia «toca / no toca» del bot debe dar exactamente
    las fechas del motor (descontada la asignación inicial, que no cuenta como rebalanceo periódico).

Se relaciona con:
    - src/ypyquant/ejecucion/calendario.py (única fuente de verdad del calendario).
    - src/ypyquant/portfolio/backtest_engine.py (usa `fechas_de_rebalanceo`) y
      pipelines/paper_trading/ (usa `toca_rebalancear`).
    - tests/test_paper_vivo.py (la parte del bot con su estado persistente) y tests/test_backtest_motor.py.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pipelines.paper_trading.estado_bot import toca_rebalancear
from ypyquant.portfolio.backtest_engine import BacktestEngine

FRECUENCIAS = ("daily", "weekly", "monthly", "quarterly")


def _fechas_motor(inicio, fin, frecuencia, dias):
    """Fechas de rebalanceo periódico del motor: programadas alineadas al primer día hábil >= fecha."""
    programadas = BacktestEngine._get_rebalance_dates(inicio, fin, frecuencia)
    return [d for d in BacktestEngine._alinear_a_dias_de_trading(programadas, list(dias)) if d != dias[0]]


def _fechas_bot(frecuencia, dias):
    """Día a día: asignación inicial el primer día y luego `toca_rebalancear` con el último rebalanceo."""
    ultimo, toca = dias[0].date(), []
    for dia in dias[1:]:
        if toca_rebalancear(frecuencia, ultimo, dia.date()):
            toca.append(dia)
            ultimo = dia.date()
    return toca


def _dias_con_festivos(semilla, inicio="2022-03-09", fin="2024-08-30", prob_festivo=0.04):
    """Días hábiles con festivos aleatorios (también semanas y meses casi enteros sin mercado)."""
    azar = np.random.default_rng(semilla)
    dias = pd.bdate_range(inicio, fin)
    return dias[azar.random(len(dias)) > prob_festivo]


@pytest.mark.parametrize("frecuencia", FRECUENCIAS)
@pytest.mark.parametrize("semilla", range(12))
def test_el_bot_dia_a_dia_coincide_con_las_fechas_del_motor(frecuencia, semilla):
    dias = _dias_con_festivos(semilla, prob_festivo=0.04 if semilla < 8 else 0.45)
    inicio, fin = dias[0].strftime("%Y-%m-%d"), dias[-1].strftime("%Y-%m-%d")
    assert _fechas_bot(frecuencia, dias) == _fechas_motor(inicio, fin, frecuencia, dias)


@pytest.mark.parametrize("frecuencia", FRECUENCIAS)
def test_paridad_cuando_el_rango_empieza_en_fin_de_semana_o_a_mitad_de_periodo(frecuencia):
    dias = pd.bdate_range("2023-01-02", "2024-12-31")
    for inicio in ("2023-01-01", "2023-01-07", "2023-02-15", "2023-03-31", "2023-06-30"):
        validos = dias[dias >= pd.Timestamp(inicio)]
        assert _fechas_bot(frecuencia, validos) == _fechas_motor(inicio, "2024-12-31", frecuencia, validos), inicio


@pytest.mark.parametrize(("frecuencia", "ultimo", "hoy", "toca"), [
    # fin de semana: ni el sábado ni el domingo abren periodo nuevo para la semana ISO anterior
    ("weekly", date(2026, 11, 20), date(2026, 11, 21), False),
    ("weekly", date(2026, 11, 20), date(2026, 11, 23), True),  # el lunes siguiente
    # festivo en lunes: el martes es el primer día de la semana (el motor lo alinea igual)
    ("weekly", date(2026, 11, 20), date(2026, 11, 24), True),
    # semana ISO que cruza el año: jueves 31-dic-2026 y viernes 1-ene-2027 son la misma semana 53
    ("weekly", date(2026, 12, 31), date(2027, 1, 1), False),
    ("weekly", date(2026, 12, 31), date(2027, 1, 4), True),
    ("monthly", date(2026, 12, 31), date(2027, 1, 4), True),  # 1-ene festivo y 2-3 fin de semana
    ("monthly", date(2026, 10, 30), date(2026, 11, 2), True),  # el 1-nov es domingo
    ("monthly", date(2026, 11, 2), date(2026, 11, 30), False),
    ("quarterly", date(2026, 3, 31), date(2026, 4, 1), True),
    ("quarterly", date(2026, 4, 1), date(2026, 6, 30), False),
    ("quarterly", date(2026, 6, 30), date(2026, 7, 1), True),
    ("daily", date(2026, 11, 20), date(2026, 11, 21), True),  # un sábado cuenta como día nuevo; el bot no corre
    ("monthly", None, date(2026, 11, 20), True),  # sin rebalanceo previo: toca ya
    ("monthly", date(2026, 11, 20), date(2026, 11, 10), False),  # reloj atrasado
    ("monthly", date(2026, 8, 3), date(2026, 11, 2), True),  # el PC estuvo apagado meses: un solo rebalanceo
])
def test_toca_rebalancear_casos_limite(frecuencia, ultimo, hoy, toca):
    assert toca_rebalancear(frecuencia, ultimo, hoy) is toca


def test_frecuencia_desconocida_falla_con_un_mensaje_claro():
    with pytest.raises(ValueError, match="[Ff]recuencia"):
        BacktestEngine._get_rebalance_dates("2023-01-01", "2023-12-31", "anual")
