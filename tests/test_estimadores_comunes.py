"""
test_estimadores_comunes.py — Tests de los parámetros `estimador_sigma` y `frecuencia_estimacion` de los
estimadores robustos (`ypyquant.estrategias.incluidas`) y de que NO cambian las huellas ya guardadas.

Se relaciona con: src/ypyquant/estrategias/incluidas.py (estimadores), .experimentos (huella),
.causalidad (verificar_sin_lookahead) y tests/test_estrategias.py (huella fija fase 0.4).
"""

import numpy as np
import pandas as pd
import pytest

from ypyquant.estrategias.base import matriz_cierres
from ypyquant.estrategias.causalidad import verificar_sin_lookahead
from ypyquant.estrategias.experimentos import ConfiguracionExperimento
from ypyquant.estrategias.incluidas import retornos_ventana_mensual
from ypyquant.estrategias.precios import CargadorSintetico
from ypyquant.estrategias.registro import cargar_estrategias, obtener_estrategia

cargar_estrategias()

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture(scope="module")
def datos() -> dict[str, pd.DataFrame]:
    cargador = CargadorSintetico()
    return {t: cargador(t, "2016-01-01", "2021-06-30") for t in TICKERS}


def _config(estrategia: str, parametros: dict) -> ConfiguracionExperimento:
    return ConfiguracionExperimento(
        estrategia=estrategia, parametros=parametros, universo=["AAPL", "MSFT", "JPM"],
        fecha_inicio="2018-01-01", fecha_fin="2024-12-31").normalizada()


# ---- huellas ---------------------------------------------------------------------------------


def test_huellas_de_estimadores_no_cambian_con_los_parametros_nuevos():
    """Calculadas con `main` (1f3b373) ANTES de añadir `estimador_sigma` y `frecuencia_estimacion`."""
    assert _config("bayes_stein", {}).huella() == "f4a0cdfc531c836331edce0961aa76222ab373d024825fff22133b75ebe83fdf"
    assert _config("bayes_stein", {"ventana_dias": 504, "peso_maximo": 0.2}).huella() == \
        "aa091e55b4410d58c920d66550af7ee902cfae05d7a32ddf6067d7aabd5a2c26"
    assert _config("michaud", {}).huella() == "5caf5126c55ac36aa948cbfbf777c8a1f8e3787c0cdcca70f1b016fc3ead9258"
    assert _config("bl_prior", {}).huella() == "0f57b83acc00c472223366599768350ea62a1e754d3d96fdc0e6e2a7379cc407"
    assert _config("robusta_elipsoide", {}).huella() == "f1e5c6075a9a2d698cd31ee83641ee564e4981bbb10d360943ef55ca5eb65adc"
    assert _config("robusta_caja", {"z": 1.5}).huella() == "6357a6915fa4dee2b652a0536f5033c571ecf479251adec63d68f5b408af296c"


def test_valores_no_defecto_si_cambian_la_huella_y_los_defectos_explicitos_no():
    base = _config("bayes_stein", {})
    assert _config("bayes_stein", {"frecuencia_estimacion": "diaria", "estimador_sigma": "ledoit_wolf"}).huella() == base.huella()
    assert _config("bayes_stein", {"frecuencia_estimacion": "mensual"}).huella() != base.huella()
    assert _config("bayes_stein", {"estimador_sigma": "muestral"}).huella() != base.huella()
    assert _config("bayes_stein", {"frecuencia_estimacion": "mensual"}).huella() != \
        _config("bayes_stein", {"estimador_sigma": "muestral"}).huella()


# ---- retornos mensuales -----------------------------------------------------------------------


def test_retornos_mensuales_descartan_el_mes_en_curso_incompleto(datos):
    recortados = {t: df.loc[:"2020-06-17"] for t, df in datos.items()}  # mitad de junio
    mensuales = retornos_ventana_mensual(recortados, 756, pd.Timestamp("2020-06-17"))
    assert mensuales.index[-1] == pd.Timestamp("2020-05-29")  # último día hábil de mayo; junio queda fuera
    cierres = matriz_cierres(recortados)
    esperado = cierres.loc["2020-05-29"] / cierres.loc["2020-04-30"] - 1.0
    np.testing.assert_allclose(mensuales.iloc[-1].values, esperado.values)


def test_retornos_mensuales_incluyen_el_mes_si_la_fecha_es_su_ultimo_dia_habil(datos):
    recortados = {t: df.loc[:"2020-06-30"] for t, df in datos.items()}
    mensuales = retornos_ventana_mensual(recortados, 756, pd.Timestamp("2020-06-30"))
    assert mensuales.index[-1] == pd.Timestamp("2020-06-30")


# ---- efecto de los parámetros -----------------------------------------------------------------


def _pesos(nombre: str, parametros: dict, datos, fecha="2020-12-31") -> dict[str, float]:
    estrategia = obtener_estrategia(nombre)(parametros)
    recortados = {t: df.loc[:fecha] for t, df in datos.items()}
    return estrategia.generar_pesos(pd.Timestamp(fecha), recortados, {})


@pytest.mark.parametrize("nombre", ["bayes_stein", "bl_prior", "robusta_elipsoide", "robusta_caja"])
def test_frecuencia_mensual_da_pesos_validos_y_distintos_a_la_diaria(nombre, datos):
    base = {"ventana_dias": 756, "peso_maximo": 0.6}
    diaria = _pesos(nombre, base, datos)
    mensual = _pesos(nombre, {**base, "frecuencia_estimacion": "mensual"}, datos)
    assert sum(mensual.values()) == pytest.approx(1.0, abs=1e-6)
    assert max(mensual.values()) <= 0.6 + 1e-6
    assert any(abs(mensual[t] - diaria[t]) > 1e-4 for t in mensual)


def test_estimador_sigma_muestral_difiere_de_ledoit_wolf(datos):
    base = {"ventana_dias": 756, "peso_maximo": 0.6}
    lw = _pesos("bayes_stein", base, datos)
    muestral = _pesos("bayes_stein", {**base, "estimador_sigma": "muestral"}, datos)
    assert sum(muestral.values()) == pytest.approx(1.0, abs=1e-6)
    assert any(abs(muestral[t] - lw[t]) > 1e-6 for t in lw)


def test_michaud_admite_frecuencia_mensual_y_sigma_muestral(datos):
    pesos = _pesos("michaud", {"n_replicas": 8, "frecuencia_estimacion": "mensual", "estimador_sigma": "muestral",
                               "peso_maximo": 0.6}, datos)
    assert sum(pesos.values()) == pytest.approx(1.0, abs=1e-6)


def test_pocas_observaciones_mensuales_avisan_y_reparten_1_sobre_n(datos):
    estrategia = obtener_estrategia("bayes_stein")({"ventana_dias": 200, "frecuencia_estimacion": "mensual"})
    recortados = {t: df.loc[:"2020-12-31"] for t, df in datos.items()}
    pesos = estrategia.generar_pesos(pd.Timestamp("2020-12-31"), recortados, {})
    assert pesos == {t: pytest.approx(0.25) for t in TICKERS}
    assert any("mensual" in aviso for aviso in estrategia.avisos)


# ---- causalidad ------------------------------------------------------------------------------


@pytest.mark.parametrize("nombre", ["bayes_stein", "robusta_caja"])
def test_variante_mensual_no_mira_al_futuro(nombre, datos):
    """Fechas a mitad de mes (mes en curso incompleto) y a fin de mes: el futuro envenenado no cambia los pesos."""
    fechas = [pd.Timestamp(f) for f in ("2020-06-17", "2020-09-30", "2020-11-10", "2020-12-31")]
    parametros = {"ventana_dias": 756, "peso_maximo": 0.6, "frecuencia_estimacion": "mensual"}
    assert verificar_sin_lookahead(obtener_estrategia(nombre), parametros, datos, fechas) == []
