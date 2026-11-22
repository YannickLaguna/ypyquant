"""
test_paper_cli.py — `python -m ypyquant.paper` (la orden que invoca el Programador de tareas de Windows).

Responsabilidad: con `YPYQUANT_PRECIOS=sintetico` y la raíz temporal `YPYQUANT_PAPER_RAIZ`, `avanzar` imprime el resumen
por sesión y sale con 0 (también si no hay nada pendiente) y con 1 si alguna sesión falló; `estado` y el control de
sesiones funcionan. Se relaciona con: src/ypyquant/paper/__main__.py y servicio.py.
"""

import pytest

from tests._paper_utiles import reloj
from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.receta import Receta
from ypyquant.paper.__main__ import main
from ypyquant.paper.almacen import AlmacenPaper
from ypyquant.paper.servicio import ServicioPaper


@pytest.fixture()
def raiz(tmp_path, monkeypatch):
    monkeypatch.setenv("YPYQUANT_PRECIOS", "sintetico")
    monkeypatch.setenv("YPYQUANT_PAPER_RAIZ", str(tmp_path / "paper"))
    return tmp_path / "paper"


def _crear_sesion(raiz, **extra):
    receta = Receta(estrategia="pesos_iguales", universo=["AAA", "BBB", "CCC"], frecuencia_rebalanceo="monthly",
                    dias_historia_previa=60).normalizada()
    return ServicioPaper(AlmacenPaper(raiz), FuenteSintetica(), ahora=reloj("2021-03-31")).crear_sesion(
        receta, fecha_alta="2021-03-01", modo_ejecucion="cierre", **extra)


def test_avanzar_resume_cada_sesion_y_sale_con_cero(raiz, capsys):
    sesion = _crear_sesion(raiz)
    assert main(["avanzar", "--hasta", "2021-03-12"]) == 0
    salida = capsys.readouterr().out
    assert sesion.id in salida and "10 día(s) hasta 2021-03-12" in salida
    assert main(["avanzar", "--hasta", "2021-03-12"]) == 0  # idempotente: ya está al día
    assert "al día" in capsys.readouterr().out


def test_sin_sesiones_no_es_un_error(raiz, capsys):
    assert main(["avanzar"]) == 0
    assert "No hay sesiones activas" in capsys.readouterr().out


def test_codigo_de_salida_distinto_de_cero_si_una_sesion_falla(raiz, capsys, monkeypatch):
    _crear_sesion(raiz)
    monkeypatch.setenv("YPYQUANT_OFFLINE", "1")
    monkeypatch.setattr("ypyquant.paper.__main__.fuente_por_defecto", lambda: _FuenteRota())
    assert main(["avanzar"]) == 1
    assert "ERROR" in capsys.readouterr().out


class _FuenteRota(FuenteSintetica):
    def ohlcv(self, *args, **kwargs):
        return None  # sin datos de ningún ticker


def test_estado_y_control_de_sesiones(raiz, capsys):
    sesion = _crear_sesion(raiz)
    assert main(["estado"]) == 0
    assert sesion.id in capsys.readouterr().out
    assert main(["pausar", sesion.id]) == 0
    assert "pausada" in capsys.readouterr().out
    assert main(["avanzar"]) == 0  # una pausada no se avanza
    assert main(["detener", sesion.id]) == 0
