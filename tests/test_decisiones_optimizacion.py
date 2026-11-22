"""
Tests de las decisiones del 2026-11-22 sobre optimización: volatilidad objetivo inalcanzable (aviso explícito en la
fachada y en los avisos del experimento) y apalancamiento como parámetro de las estrategias con volatilidad objetivo.

Se relaciona con: ypyquant.optimization.optimizador_portafolio (fachada), ypyquant.estrategias.incluidas
(estrategias), ypyquant.estrategias.base (`sanear_pesos`), ypyquant.portfolio.backtest_engine (margen).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, ejecutar_experimento
from ypyquant.optimization import optimizador_portafolio as fachada

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
ESTRATEGIAS_CON_VOLATILIDAD = ["bayes_stein", "bl_prior", "michaud", "robusta_elipsoide", "robusta_caja"]


def _config(**cambios) -> ConfiguracionExperimento:
    base = {"estrategia": "bayes_stein", "universo": TICKERS, "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}
    return ConfiguracionExperimento(**{**base, **cambios})


def _parametros(estrategia: str, **extra) -> dict:
    return {**({"n_replicas": 6} if estrategia == "michaud" else {}), **extra}


def _retornos(semilla: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    return pd.DataFrame(rng.normal(0.0004, 0.01, size=(500, 4)), columns=list("ABCD"))


# ---- tarea 1: volatilidad objetivo inalcanzable -> mínima varianza, y se dice ---------------------


def test_fachada_vol_objetivo_inalcanzable_trae_aviso_legible():
    res = fachada.optimizar_volatilidad_objetivo(_retornos(), 0.001)
    assert res["estado"] == "gmv_volatilidad_inalcanzable" and res["volatility_achieved"] is False
    esperado = f"σ objetivo 0,1 % < σ mínima alcanzable {res['volatility'] * 100:.1f}".replace(".", ",")
    assert res["aviso"].startswith(esperado) and res["aviso"].endswith("se usa la cartera de mínima varianza")


def test_fachada_vol_objetivo_alcanzable_no_trae_aviso():
    res = fachada.optimizar_volatilidad_objetivo(_retornos(), 0.15)
    assert res["aviso"] is None


@pytest.mark.parametrize("estrategia", ESTRATEGIAS_CON_VOLATILIDAD)
def test_experimento_avisa_agregado_cuando_la_vol_objetivo_es_inalcanzable(estrategia):
    resultado = ejecutar_experimento(
        _config(estrategia=estrategia, parametros=_parametros(estrategia, volatilidad_objetivo=0.001)),
        FuenteSintetica())
    avisos = [a for a in resultado.avisos if "mínima varianza" in a]
    assert len(avisos) == 1, resultado.avisos  # agregado: uno solo aunque haya muchos rebalanceos
    n_rebalanceos = resultado.metricas["n_rebalanceos"]
    assert n_rebalanceos > 3
    assert "σ objetivo 0,1 %" in avisos[0] and "σ mínima alcanzable" in avisos[0]
    assert f"de {n_rebalanceos} rebalanceos" in avisos[0] or "réplicas" in avisos[0]
    assert "2020-" in avisos[0] or "2021-" in avisos[0]  # fecha de rebalanceo (primero/último)


@pytest.mark.parametrize("estrategia", ESTRATEGIAS_CON_VOLATILIDAD)
def test_experimento_sin_vol_inalcanzable_no_avisa(estrategia):
    resultado = ejecutar_experimento(
        _config(estrategia=estrategia, parametros=_parametros(estrategia, volatilidad_objetivo=0.9)), FuenteSintetica())
    assert not [a for a in resultado.avisos if "mínima varianza" in a]


def test_el_aviso_llega_al_detalle_guardado_del_experimento(tmp_path):
    """Camino de la API/web: el aviso agregado se guarda en `metricas.json` y el detalle lo devuelve en `metricas_completas`."""
    from ypyquant.estrategias.almacen import AlmacenExperimentos

    almacen = AlmacenExperimentos(tmp_path)
    config = _config(parametros={"volatilidad_objetivo": 0.001}).normalizada()
    id_experimento = almacen.registrar(config)
    almacen.guardar(id_experimento, ejecutar_experimento(config, FuenteSintetica()))
    avisos = almacen.leer(id_experimento)["metricas_completas"]["avisos"]
    assert any("Volatilidad objetivo inalcanzable" in a and "mínima varianza" in a for a in avisos)
