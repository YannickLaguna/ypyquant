"""
Tests de `Receta`: la canonización única de la huella y la composición receta + periodo = experimento.

Responsabilidad: fija valores de huella calculados con el código ANTERIOR a la refactorización (no deben cambiar) y
comprueba que receta, experimento, configuración guardada y marco comparten la misma canonización.
Se relaciona con: ypyquant.estrategias.receta, .experimentos, .configuraciones y .marcos;
tests/test_huellas_guardadas.py (lo mismo contra los datos reales).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ypyquant.estrategias.configuraciones import ConfiguracionGuardada
from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.estrategias.marcos import MarcoComparacion, aplicar_marco
from ypyquant.estrategias.receta import Receta, valor_comparable

# Calculadas con la implementación previa (huella_receta / huella con fechas falsas)
HUELLA_EXPERIMENTO = "5704d391e68a1d87120a5ecbd6964efcb65fee7f470c50c6bdbc2004b5988c40"
HUELLA_RECETA = "33e6d932fc5badcb565a772ba7009d989b9cffc4932c2c14da9201029fcb1a74"


def _experimento(**cambios) -> ConfiguracionExperimento:
    base = dict(estrategia="momentum", parametros={"top_k": 3}, universo=["b", "a", "c"],
                fecha_inicio="2020-01-01", fecha_fin="2021-01-01")
    return ConfiguracionExperimento(**{**base, **cambios}).normalizada()


def test_huellas_fijas_no_cambian():
    experimento = _experimento()
    assert experimento.huella() == HUELLA_EXPERIMENTO
    assert experimento.huella_receta() == HUELLA_RECETA
    assert experimento.receta().huella() == HUELLA_RECETA


def test_la_receta_ignora_periodo_capital_validaciones_y_etiqueta():
    otro = _experimento(fecha_inicio="2018-01-01", fecha_fin="2019-06-01", capital_inicial=5_000.0,
                        validaciones=["psr"], etiqueta="x")
    assert otro.huella() != HUELLA_EXPERIMENTO and otro.huella_receta() == HUELLA_RECETA


def test_la_huella_no_depende_del_orden_del_universo_ni_de_la_etiqueta():
    assert _experimento(universo=["C", "A", "B"], etiqueta="otra").huella() == HUELLA_EXPERIMENTO


def test_receta_sola_normaliza_y_da_la_misma_huella():
    receta = Receta(estrategia="momentum", parametros={"top_k": 3}, universo=["b", "a", "c"])
    assert receta.normalizada().huella() == HUELLA_RECETA
    assert receta.normalizada().frecuencia_rebalanceo is not None and receta.frecuencia_rebalanceo is None


def test_desde_receta_compone_el_experimento_con_la_misma_huella():
    experimento = _experimento()
    reconstruido = ConfiguracionExperimento.desde_receta(
        experimento.receta(), experimento.fecha_inicio, experimento.fecha_fin, experimento.capital_inicial).normalizada()
    assert reconstruido.huella() == experimento.huella()
    assert type(experimento.receta()) is Receta


def test_receta_valida_universo_vacio_y_campos_extra():
    with pytest.raises(ValidationError):
        Receta(estrategia="momentum", universo=[])
    with pytest.raises(ValidationError):
        Receta(estrategia="momentum", universo=["A"], fecha_inicio="2020-01-01")


def test_valor_comparable_ignora_el_orden_de_las_listas():
    assert valor_comparable(["B", "A"]) == valor_comparable(["A", "B"]) != valor_comparable(["A"])
    assert valor_comparable({"b": 1, "a": 2}) == valor_comparable({"a": 2, "b": 1})


def test_guardada_y_marco_componen_la_misma_receta():
    experimento = _experimento()
    guardada = ConfiguracionGuardada.desde_experimento(experimento, nombre="Mi receta")
    assert guardada.huella_estrategia == HUELLA_RECETA == guardada.receta().huella()
    marco = MarcoComparacion(nombre="m", universo=guardada.universo, fecha_inicio="2020-01-01", fecha_fin="2021-01-01",
                             **{k: v for k, v in guardada.entorno().items() if k != "universo"})
    config, _ = aplicar_marco(marco, "momentum", {"top_k": 3})
    assert config.huella_receta() == HUELLA_RECETA and config.huella() == HUELLA_EXPERIMENTO
