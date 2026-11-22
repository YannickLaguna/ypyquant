"""
Test de contrato de la interfaz `FuentePrecios`: el mismo conjunto de pruebas corre contra TODOS los adaptadores.

Responsabilidad:
- Fija lo que cualquier fuente debe cumplir (formato OHLCV, ajustado frente a crudo, identificador estable,
  deslistados con fecha, símbolo desconocido -> None, compatibilidad con el contrato heredado `CargadorPrecios`).
  Un adaptador nuevo (Norgate) entra añadiendo una fábrica a `FABRICAS`: si pasa estas pruebas, el resto del
  sistema no cambia.
- Los adaptadores se construyen sin red: el sintético es determinista y el de yfinance se precarga en una
  caché Parquet temporal con `descargar=False` (así se ejercita su lógica de caché y de inferencia).
- Además prueba los deslistados inferidos de `FuenteDesdeCargador` y `como_fuente`.

Se relaciona con: ypyquant.data.fuente_precios, ypyquant.data.fuentes (sintetica, yfinance_cache).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from ypyquant.data.fuente_precios import (
    COLUMNAS_OHLCV,
    Deslistado,
    FuenteDesdeCargador,
    FuentePrecios,
    como_fuente,
    inferir_deslistado,
)
from ypyquant.data.fuentes import FuenteSintetica, FuenteYFinanceCache

INICIO, FIN = "2019-01-02", "2020-12-31"
FECHA_DESLISTADO = "2020-07-03"


@dataclass
class Caso:
    """Una fuente lista para el contrato: un símbolo vivo, uno deslistado (en FECHA_DESLISTADO) y uno inexistente."""

    fuente: FuentePrecios
    vivo: str = "VIVO"
    muerto: str = "MUERTO"
    inexistente: str = "NOEXISTE"


def _fabrica_sintetica(tmp_path) -> Caso:
    return Caso(FuenteSintetica(deslistados={"MUERTO": (FECHA_DESLISTADO, "quiebra")}, ausentes={"NOEXISTE"}))


def _fabrica_yfinance_con_cache(tmp_path) -> Caso:
    """yfinance sin red: la caché se precarga con datos sintéticos (el muerto acaba antes del final cubierto)."""
    origen = FuenteSintetica(deslistados={"MUERTO": (FECHA_DESLISTADO, None)})
    fuente = FuenteYFinanceCache(tmp_path, descargar=False)
    for simbolo in ("VIVO", "MUERTO"):
        for ajuste in ("ajustado", "crudo"):
            fuente._guardar_cache(simbolo, origen.ohlcv(simbolo, INICIO, FIN, ajuste), INICIO, FIN, ajuste)
    return Caso(fuente)


FABRICAS = {"sintetica": _fabrica_sintetica, "yfinance_cache": _fabrica_yfinance_con_cache}


@pytest.fixture(params=sorted(FABRICAS))
def caso(request, tmp_path) -> Caso:
    return FABRICAS[request.param](tmp_path)


def test_ajustada_tiene_el_formato_estandar(caso):
    df = caso.fuente.ohlcv(caso.vivo, INICIO, FIN, "ajustado")
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert set(COLUMNAS_OHLCV) <= set(df.columns)
    assert df.index.min() >= pd.Timestamp(INICIO) and df.index.max() <= pd.Timestamp(FIN)
    assert np.isfinite(df["Close"]).all() and (df["Close"] > 0).all()
    assert df["Close"].equals(df["adjusted_close"])


def test_cruda_comparte_indice_y_conserva_el_ajustado(caso):
    ajustada = caso.fuente.ohlcv(caso.vivo, INICIO, FIN, "ajustado")
    cruda = caso.fuente.ohlcv(caso.vivo, INICIO, FIN, "crudo")
    assert cruda.index.equals(ajustada.index)
    assert set(COLUMNAS_OHLCV) <= set(cruda.columns)
    assert (cruda["Close"] > 0).all()
    np.testing.assert_allclose(cruda["adjusted_close"], ajustada["adjusted_close"])
    assert not cruda["Close"].equals(cruda["adjusted_close"])  # la serie cruda es de verdad distinta de la ajustada


def test_el_rango_pedido_se_respeta(caso):
    df = caso.fuente.ohlcv(caso.vivo, "2020-03-02", "2020-06-30")
    assert df.index.min() >= pd.Timestamp("2020-03-02") and df.index.max() <= pd.Timestamp("2020-06-30")
    assert len(df) > 50


def test_simbolo_desconocido_devuelve_none(caso):
    assert caso.fuente.ohlcv(caso.inexistente, INICIO, FIN) is None
    assert caso.fuente.deslistado(caso.inexistente) is None


def test_llamable_como_cargador_precios_heredado(caso):
    esperado = caso.fuente.ohlcv(caso.vivo, INICIO, FIN, "ajustado")
    pd.testing.assert_frame_equal(caso.fuente(caso.vivo, INICIO, FIN), esperado)


def test_identificador_estable_y_distinto_por_valor(caso):
    f = caso.fuente
    assert isinstance(f.identificador(caso.vivo), str) and f.identificador(caso.vivo)
    assert f.identificador(caso.vivo) == f.identificador(caso.vivo)
    assert f.identificador(caso.vivo) != f.identificador(caso.muerto)


def test_el_valor_vivo_no_figura_como_deslistado(caso):
    caso.fuente.ohlcv(caso.vivo, INICIO, FIN)
    assert caso.fuente.deslistado(caso.vivo) is None


def test_el_deslistado_trae_su_fecha_y_la_serie_no_la_supera(caso):
    df = caso.fuente.ohlcv(caso.muerto, INICIO, FIN)
    info = caso.fuente.deslistado(caso.muerto)
    assert isinstance(info, Deslistado)
    assert info.fecha == pd.Timestamp(FECHA_DESLISTADO) == df.index.max()


# ---- adaptadores concretos: lo que NO es contrato común -----------------------------------------------


def test_sintetica_declara_el_motivo_del_deslistado_como_dato_explicito():
    fuente = FuenteSintetica(deslistados={"XXX": (FECHA_DESLISTADO, "quiebra")})
    info = fuente.deslistado("XXX")
    assert info.motivo == "quiebra" and info.inferido is False


def test_yfinance_infiere_el_deslistado_sin_motivo(tmp_path):
    info = _fabrica_yfinance_con_cache(tmp_path).fuente.deslistado("MUERTO")
    assert info.motivo is None and info.inferido is True


# ---- cargador suelto y regla de inferencia -------------------------------------------------------------


def test_cargador_suelto_se_envuelve_y_infiere_deslistados():
    sintetica = FuenteSintetica(deslistados={"MUERTO": (FECHA_DESLISTADO, None)})
    fuente = como_fuente(lambda t, i, f: sintetica.ohlcv(t, i, f))
    assert isinstance(fuente, FuenteDesdeCargador) and como_fuente(fuente) is fuente
    fuente.ohlcv("VIVO", INICIO, FIN)
    fuente.ohlcv("MUERTO", INICIO, FIN)
    assert fuente.deslistado("VIVO") is None
    assert fuente.deslistado("MUERTO") == Deslistado(pd.Timestamp(FECHA_DESLISTADO), None, True)
    with pytest.raises(ValueError):
        fuente.ohlcv("VIVO", INICIO, FIN, "crudo")


def test_regla_de_inferencia_tolera_fines_de_semana():
    assert inferir_deslistado(pd.Timestamp("2020-12-24"), pd.Timestamp("2020-12-31")) is not None
    assert inferir_deslistado(pd.Timestamp("2020-12-28"), pd.Timestamp("2020-12-31")) is None
