"""
test_fuente_refresco.py — `FuenteYFinanceCache.refrescar_reciente`: añade sesiones nuevas sin tocar lo cacheado.

Responsabilidad: sin cambios en el solape solo se AÑADEN las filas nuevas (los precios cacheados no cambian); si el
solape difiere (dividendo o split: yfinance reescribe el histórico ajustado) se vuelve a descargar el símbolo entero y se
devuelve como cambiado. Sin descargas habilitadas no hace nada. Todo sin red (la descarga se sustituye).

Se relaciona con: src/ypyquant/data/fuentes/yfinance_cache.py y src/ypyquant/paper/simulador.py (lo llama).
"""

import pandas as pd

from tests._paper_utiles import barras
from ypyquant.data.fuentes.yfinance_cache import FuenteYFinanceCache


def _fuente(tmp_path, completo: pd.DataFrame, descargas: list):
    fuente = FuenteYFinanceCache(tmp_path, descargar=True)

    def descargar(simbolo, inicio, fin, ajuste="ajustado"):
        descargas.append((inicio, fin))
        return completo.loc[inicio:fin]

    fuente._descargar = descargar  # type: ignore[method-assign]
    return fuente


def test_solo_anade_las_filas_nuevas_si_el_solape_coincide(tmp_path):
    completo = barras("2021-01-01", "2021-03-31", 100)
    cache = completo.loc[:"2021-02-26"]
    descargas: list = []
    fuente = _fuente(tmp_path, completo, descargas)
    fuente._guardar_cache("AAA", cache, "2021-01-01", "2021-02-26")
    assert fuente.refrescar_reciente(["AAA"], "2021-03-05") == []
    ampliada, cubierto = fuente._leer_cache("AAA")
    assert cubierto == ("2021-01-01", "2021-03-05") and ampliada.index[-1] == pd.Timestamp("2021-03-05")
    pd.testing.assert_frame_equal(ampliada.loc[:"2021-02-26"], cache, check_freq=False)  # lo cacheado, intacto
    assert len(descargas) == 1 and descargas[0][0] > "2021-01-01"  # solo se pidió el tramo reciente


def test_un_ajuste_en_el_solape_vuelve_a_descargar_el_simbolo(tmp_path):
    completo = barras("2021-01-01", "2021-03-31", 100)
    cache = completo.loc[:"2021-02-26"].copy()
    reescalado = completo.copy()
    reescalado.loc[:"2021-03-01", ["Open", "High", "Low", "Close", "adjusted_close"]] *= 0.98  # dividendo
    descargas: list = []
    fuente = _fuente(tmp_path, reescalado, descargas)
    fuente._guardar_cache("AAA", cache, "2021-01-01", "2021-02-26")
    assert fuente.refrescar_reciente(["AAA"], "2021-03-05") == ["AAA"]
    nuevo, _ = fuente._leer_cache("AAA")
    assert nuevo["Close"].iloc[0] == reescalado["Close"].iloc[0]  # historial sustituido por el nuevo ajuste
    assert len(descargas) == 2  # tramo reciente + descarga completa


def test_sin_descargas_o_sin_cache_no_hace_nada(tmp_path):
    descargas: list = []
    fuente = _fuente(tmp_path, barras("2021-01-01", "2021-03-31", 100), descargas)
    assert fuente.refrescar_reciente(["AAA"], "2021-03-05") == [] and descargas == []  # sin caché previa
    apagada = FuenteYFinanceCache(tmp_path, descargar=False)
    assert apagada.refrescar_reciente(["AAA"], "2021-03-05") == []
