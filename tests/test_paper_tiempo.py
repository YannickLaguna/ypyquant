"""
test_paper_tiempo.py — Cuánto tiempo lleva una sesión de paper interno y cuándo toca su próximo rebalanceo.

Responsabilidad:
    - `proximo_rebalanceo` (función pura de `paper/calendario.py`): la fecha del siguiente rebalanceo según la frecuencia
      de la receta, el último rebalanceo y la última fecha procesada (semanal, mensual, trimestral, diario; fin de semana;
      sin rebalanceo previo; rebalanceo ya vencido);
    - el resumen y el detalle de una sesión traen lo necesario para explicar el tiempo en la web: periodo (alta y última
      fecha procesada), sesiones de mercado, días pendientes, frecuencia y próxima fecha de rebalanceo.

Se relaciona con: ypyquant.paper.calendario, ypyquant.paper.servicio y la web (lib/paperInterno.ts).
"""

from datetime import date

import pytest

from tests._paper_utiles import FuenteManual, barras, reloj
from ypyquant.estrategias.receta import Receta
from ypyquant.paper.almacen import AlmacenPaper
from ypyquant.paper.calendario import proximo_rebalanceo
from ypyquant.paper.servicio import ServicioPaper


@pytest.mark.parametrize(("frecuencia", "ultimo_rebalanceo", "ultima_fecha", "esperado"), [
    ("monthly", date(2021, 3, 1), date(2021, 3, 12), date(2021, 4, 1)),
    ("monthly", date(2021, 5, 3), date(2021, 5, 31), date(2021, 6, 1)),  # el día 1 siguiente es el mismo día siguiente
    ("monthly", date(2021, 5, 3), date(2021, 6, 1), date(2021, 6, 2)),  # el del 1 de junio ya se procesó: se ejecuta en la próxima sesión
    ("weekly", date(2021, 3, 8), date(2021, 3, 10), date(2021, 3, 15)),  # lunes siguiente
    ("quarterly", date(2021, 1, 4), date(2021, 3, 12), date(2021, 4, 1)),  # 1 de abril (jueves)
    ("daily", date(2021, 3, 12), date(2021, 3, 12), date(2021, 3, 15)),  # viernes -> lunes
    ("monthly", date(2021, 5, 3), date(2021, 5, 28), date(2021, 6, 1)),
    ("monthly", None, date(2021, 3, 4), date(2021, 3, 5)),  # sin rebalanceo previo: la próxima sesión
    ("monthly", date(2021, 1, 4), date(2021, 3, 12), date(2021, 3, 15)),  # vencido (parón largo): la próxima sesión recupera
])
def test_proximo_rebalanceo(frecuencia, ultimo_rebalanceo, ultima_fecha, esperado):
    assert proximo_rebalanceo(frecuencia, ultimo_rebalanceo, ultima_fecha) == esperado


def test_si_la_fecha_programada_cae_en_fin_de_semana_se_lleva_al_lunes():
    # 1 de mayo de 2021 es sábado: el rebalanceo mensual se ejecuta el lunes 3
    assert proximo_rebalanceo("monthly", date(2021, 4, 1), date(2021, 4, 30)) == date(2021, 5, 3)


def _servicio(tmp_path, dia="2021-03-12"):
    series = {t: barras("2019-01-01", "2021-04-30", 50 + 25 * i) for i, t in enumerate(["AAA", "BBB", "SPY"])}
    return ServicioPaper(AlmacenPaper(tmp_path / "paper"), FuenteManual(series), ahora=reloj(dia))


def _receta(frecuencia="monthly") -> Receta:
    return Receta(estrategia="pesos_iguales", parametros={}, universo=["AAA", "BBB"], frecuencia_rebalanceo=frecuencia,
                  comision=0.001, slippage=0.001, dias_historia_previa=60).normalizada()


def test_el_resumen_explica_el_tiempo_de_la_sesion(tmp_path):
    servicio = _servicio(tmp_path)
    sesion = servicio.crear_sesion(_receta(), fecha_alta="2021-03-01", modo_ejecucion="cierre")
    antes = servicio.resumen(sesion.id)
    assert antes["dias"] == 0 and antes["ultima_fecha"] is None and antes["dias_pendientes"] == 10
    assert antes["frecuencia_rebalanceo"] == "monthly" and antes["proximo_rebalanceo"] == "2021-03-01"
    servicio.avanzar()
    despues = servicio.resumen(sesion.id)
    assert despues["fecha_alta"] == "2021-03-01" and despues["ultima_fecha"] == "2021-03-12" and despues["dias"] == 10
    assert despues["dias_pendientes"] == 0 and despues["al_dia"] is True
    assert despues["ultimo_rebalanceo"] == "2021-03-01" and despues["proximo_rebalanceo"] == "2021-04-01"
    assert servicio.detalle(sesion.id)["proximo_rebalanceo"] == "2021-04-01"
