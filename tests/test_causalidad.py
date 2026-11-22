"""
test_causalidad.py — Tests de `verificar_sin_lookahead` (ypyquant.estrategias.causalidad).

Responsabilidad: comprobar que la utilidad acepta estrategias causales (las incluidas y una escrita a
mano) y detecta las que miran al futuro en `preparar()` (`shift(-1)`, `rolling(center=True)`), y que el
mensaje de error cumple el formato del plan (`look-ahead en <fecha>: <ticker> pesa X con futuro real y
Y con futuro envenenado`). Datos sintéticos deterministas, sin red.

Se relaciona con: src/ypyquant/estrategias/causalidad.py, base.py (Estrategia.como_funcion) e
incluidas.py (estrategias que deben salir limpias).
"""

import pandas as pd
import pytest

from ypyquant.estrategias.base import Estrategia, ParametrosBase, matriz_cierres
from ypyquant.estrategias.causalidad import envenenar_futuro, verificar_sin_lookahead
from ypyquant.data.fuentes import FuenteSintetica

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
FECHAS_MUESTRA = [pd.Timestamp(f) for f in ("2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31")]


@pytest.fixture(scope="module")
def datos() -> dict[str, pd.DataFrame]:
    cargador = FuenteSintetica()
    return {t: cargador(t, "2018-01-01", "2021-06-30") for t in TICKERS}


def _mejor(serie: pd.Series) -> dict[str, float]:
    """Todo el peso en el ticker con mayor valor de `serie` (ignora NaN)."""
    serie = serie.dropna()
    return {serie.idxmax(): 1.0} if not serie.empty else {}


class Limpia(Estrategia):
    """Causal: momentum de 60 sesiones calculado en `generar_pesos` con datos ya recortados."""

    nombre = "limpia_test"

    def generar_pesos(self, fecha, datos, estado):
        cierres = matriz_cierres(datos)
        return _mejor(cierres.iloc[-1] / cierres.iloc[-61] - 1.0)


class LimpiaConPreparar(Estrategia):
    """Causal: precalcula en `preparar` un indicador solo hacia atrás (`rolling` por defecto)."""

    nombre = "limpia_preparar_test"

    def preparar(self, datos):
        self.media = matriz_cierres(datos).rolling(20).mean()

    def generar_pesos(self, fecha, datos, estado):
        cierres = matriz_cierres(datos)
        return _mejor(cierres.iloc[-1] / self.media.loc[cierres.index[-1]] - 1.0)


class ConShiftNegativo(Estrategia):
    """Look-ahead: `preparar` calcula el retorno del día SIGUIENTE (`shift(-1)`) y elige al mejor."""

    nombre = "shift_test"

    def preparar(self, datos):
        self.retorno_siguiente = matriz_cierres(datos).pct_change().shift(-1)

    def generar_pesos(self, fecha, datos, estado):
        return _mejor(self.retorno_siguiente.loc[pd.Timestamp(fecha)])


class ConVentanaCentrada(Estrategia):
    """Look-ahead: media móvil centrada (`rolling(center=True)`) usa 2 días posteriores a la fecha."""

    nombre = "centrada_test"

    def preparar(self, datos):
        cierres = matriz_cierres(datos)
        self.suavizada = cierres.rolling(5, center=True).mean()
        self.cierres = cierres

    def generar_pesos(self, fecha, datos, estado):
        fila = pd.Timestamp(fecha)
        return _mejor(self.suavizada.loc[fila] / self.cierres.loc[fila])


class ConParametros(Estrategia):
    """Causal y con parámetros: comprueba que `verificar_sin_lookahead` los pasa a la clase."""

    nombre = "con_parametros_test"

    class Parametros(ParametrosBase):
        ventana: int = 30

    def generar_pesos(self, fecha, datos, estado):
        cierres = matriz_cierres(datos)
        return _mejor(cierres.iloc[-1] / cierres.iloc[-(self.parametros.ventana + 1)] - 1.0)


def test_estrategia_limpia_no_da_errores(datos):
    assert verificar_sin_lookahead(Limpia, {}, datos, FECHAS_MUESTRA) == []


def test_estrategia_limpia_con_preparar_causal_no_da_errores(datos):
    assert verificar_sin_lookahead(LimpiaConPreparar, {}, datos, FECHAS_MUESTRA) == []


def test_parametros_se_pasan_a_la_clase(datos):
    assert verificar_sin_lookahead(ConParametros, {"ventana": 10}, datos, FECHAS_MUESTRA) == []


def test_shift_negativo_en_preparar_se_detecta(datos):
    errores = verificar_sin_lookahead(ConShiftNegativo, {}, datos, FECHAS_MUESTRA)
    assert errores, "shift(-1) en preparar() debe detectarse"
    assert all(e.startswith("look-ahead en ") for e in errores)
    assert "con futuro real y" in errores[0] and "con futuro envenenado" in errores[0]


def test_rolling_centrado_en_preparar_se_detecta(datos):
    assert verificar_sin_lookahead(ConVentanaCentrada, {}, datos, FECHAS_MUESTRA)


def test_formato_del_mensaje_incluye_fecha_ticker_y_pesos(datos):
    error = verificar_sin_lookahead(ConShiftNegativo, {}, datos, FECHAS_MUESTRA)[0]
    assert error.startswith("look-ahead en 2020-")
    assert any(f" {t} pesa " in error for t in TICKERS)


def test_es_determinista_con_la_misma_semilla(datos):
    a = verificar_sin_lookahead(ConShiftNegativo, {}, datos, FECHAS_MUESTRA, semilla=3)
    b = verificar_sin_lookahead(ConShiftNegativo, {}, datos, FECHAS_MUESTRA, semilla=3)
    assert a == b


def test_no_modifica_los_datos_de_entrada(datos):
    copia = {t: df.copy() for t, df in datos.items()}
    verificar_sin_lookahead(ConShiftNegativo, {}, datos, FECHAS_MUESTRA)
    for t in datos:
        pd.testing.assert_frame_equal(datos[t], copia[t])


def test_las_estrategias_incluidas_causales_pasan(datos):
    from ypyquant.estrategias.registro import cargar_estrategias, obtener_estrategia

    cargar_estrategias()
    for nombre, parametros in (("pesos_iguales", {}), ("momentum", {"top_k": 2, "meses_retorno": 6})):
        assert verificar_sin_lookahead(obtener_estrategia(nombre), parametros, datos, FECHAS_MUESTRA) == [], nombre


def test_envenenar_futuro_solo_cambia_lo_posterior_a_la_fecha(datos):
    fecha = pd.Timestamp("2020-06-30")
    for modo in ("ruido", "nan"):
        veneno = envenenar_futuro(datos, fecha, semilla=0, modo=modo)
        for t, df in datos.items():
            pd.testing.assert_frame_equal(veneno[t].loc[:fecha], df.loc[:fecha])
            assert list(veneno[t].index) == list(df.index)  # mismo calendario
            posterior = veneno[t].loc[fecha + pd.Timedelta(days=1):]
            if modo == "nan":
                assert posterior.isna().all().all()
            else:
                assert (posterior["Close"] != df.loc[posterior.index, "Close"]).all()
                assert (posterior["Close"] > 0).all()
