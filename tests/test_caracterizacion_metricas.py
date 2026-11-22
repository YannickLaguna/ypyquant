"""
test_caracterizacion_metricas.py — Fija TODAS las métricas de varios experimentos sintéticos.

Responsabilidad:
    Test de caracterización escrito ANTES de partir `ejecutar_con_datos` en `simular` + `evaluar` y de
    quitar la semilla global: los valores de `tests/datos/caracterizacion_metricas.json` salen del código
    previo (todas las claves de `metricas`, tamaños y suma de la equity). Cubre una estrategia determinista
    con validadores (momentum), una con aleatoriedad (michaud: bootstrap con generador propio) y una de
    pesos fijos con rebalanceo semanal. Si algo cambia, el refactor ha alterado resultados guardados.
    Además comprueba que el resultado no depende del estado global de numpy.random ni de `random`.

Se relaciona con:
    - src/ypyquant/estrategias/experimentos.py (`ejecutar_experimento`), .simulacion y .evaluacion.
    - tests/test_estrategias.py (huella fija) y tests/test_simular_evaluar.py (nueva interfaz).
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from ypyquant.estrategias.experimentos import ConfiguracionExperimento, ejecutar_experimento
from ypyquant.estrategias.precios import CargadorSintetico

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
CASOS = {
    "momentum": {"estrategia": "momentum", "parametros": {"top_k": 3, "meses_retorno": 6}, "validaciones": ["rachas", "psr"]},
    "michaud": {"estrategia": "michaud", "parametros": {"n_replicas": 8}, "semilla": 3, "validaciones": ["psr"]},
    "pesos_iguales": {"estrategia": "pesos_iguales", "frecuencia_rebalanceo": "weekly"},
}
ESPERADO = json.loads((Path(__file__).parent / "datos" / "caracterizacion_metricas.json").read_text(encoding="utf-8"))


def _ejecutar(nombre: str) -> dict:
    config = ConfiguracionExperimento(universo=TICKERS, fecha_inicio="2020-01-01", fecha_fin="2021-12-31",
                                      **CASOS[nombre])
    r = ejecutar_experimento(config, CargadorSintetico())
    return {nombre: r.metricas,
            nombre + "_n": [len(r.equity), len(r.pesos), len(r.operaciones), float(r.equity.equity.sum()), r.avisos]}


@pytest.mark.parametrize("nombre", list(CASOS))
def test_metricas_completas_identicas_al_codigo_previo(nombre):
    obtenido = json.loads(json.dumps(_ejecutar(nombre), sort_keys=True))
    assert obtenido[nombre] == ESPERADO[nombre]
    assert obtenido[nombre + "_n"] == ESPERADO[nombre + "_n"]


@pytest.mark.parametrize("nombre", list(CASOS))
def test_resultado_independiente_del_estado_global_de_azar(nombre):
    random.seed(999)
    np.random.seed(999)
    obtenido = json.loads(json.dumps(_ejecutar(nombre), sort_keys=True))
    assert obtenido[nombre] == ESPERADO[nombre]
