"""
Tests de la librería de validación estadística `ypyquant.validacion`.

Responsabilidad:
- Verificar estadísticos (Sharpe, SR máximo esperado, PSR, DSR, PBO-CSCV, rachas, Ulcer/Martin,
  p-valor de permutación), la permutación de precios (método de Masters) y la selección walk-forward.

Se relaciona con:
- ypyquant.validacion (estadisticos, permutacion, walk_forward). Sin red ni datos externos: todo
  con series sintéticas y semillas fijas.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ypyquant.validacion import (
    indice_ulcer,
    p_valor_permutacion,
    pbo_cscv,
    permutar_precios,
    prueba_rachas,
    ratio_martin,
    seleccion_walk_forward,
    sharpe,
    sharpe_deflactado,
    sharpe_maximo_esperado,
    sharpe_probabilistico,
)

# --------------------------------------------------------------------------- utilidades


def _ohlcv_sintetico(semilla: int, n: int = 300, minuscula: bool = False, ajustado: str | None = None,
                     fechas: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """OHLCV de paseo aleatorio con alto >= max(ap, cierre) y bajo <= min(ap, cierre)."""
    rng = np.random.default_rng(semilla)
    indice = fechas if fechas is not None else pd.bdate_range("2020-01-01", periods=n)
    n = len(indice)
    cierre = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    apertura = np.empty(n)
    apertura[0] = cierre[0]
    apertura[1:] = cierre[:-1] * np.exp(rng.normal(0, 0.004, n - 1))
    alto = np.maximum(apertura, cierre) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    bajo = np.minimum(apertura, cierre) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    volumen = rng.integers(1_000, 100_000, n)
    nombres = ("open", "high", "low", "close", "volume") if minuscula else ("Open", "High", "Low", "Close", "Volume")
    df = pd.DataFrame(dict(zip(nombres, (apertura, alto, bajo, cierre, volumen), strict=True)), index=indice)
    if ajustado:
        factor = np.linspace(0.9, 1.0, n)
        df[ajustado] = cierre * factor
    return df


def _col(df: pd.DataFrame, nombre: str) -> pd.Series:
    return df[next(c for c in df.columns if str(c).lower() == nombre)]


def _logret(serie: pd.Series) -> np.ndarray:
    return np.diff(np.log(serie.to_numpy(float)))


# --------------------------------------------------------------------------- estadísticos


def test_sharpe_basico_y_anualizado():
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.01])
    esperado = r.mean() / r.std(ddof=1)
    assert sharpe(r) == pytest.approx(esperado)
    assert sharpe(r, periodos_anio=252, anualizar=True) == pytest.approx(esperado * math.sqrt(252))
    assert sharpe(pd.Series(r)) == pytest.approx(esperado)
    assert math.isnan(sharpe(np.zeros(10)))


def test_sharpe_dataframe_devuelve_serie_por_columna():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.01, (100, 3)), columns=["a", "b", "c"])
    resultado = sharpe(df)
    assert list(resultado.index) == ["a", "b", "c"]
    assert resultado["b"] == pytest.approx(sharpe(df["b"]))


def test_sharpe_maximo_esperado_valor_conocido_y_simulacion():
    assert sharpe_maximo_esperado(1.0, 1) == 0.0
    assert sharpe_maximo_esperado(1.0, 10) == pytest.approx(1.574, abs=0.005)
    # Comparación con la esperanza del máximo de N normales simuladas
    rng = np.random.default_rng(1)
    maximos = rng.standard_normal((20000, 10)).max(axis=1)
    assert sharpe_maximo_esperado(1.0, 10) == pytest.approx(maximos.mean(), abs=0.05)
    # Escala con sqrt(V)
    assert sharpe_maximo_esperado(4.0, 10) == pytest.approx(2 * sharpe_maximo_esperado(1.0, 10))


def test_sharpe_maximo_esperado_crece_con_n():
    valores = [sharpe_maximo_esperado(1.0, n) for n in (2, 10, 100, 1000)]
    assert valores == sorted(valores)


def test_psr_valores_extremos_y_formula():
    rng = np.random.default_rng(2)
    bueno = rng.normal(0.002, 0.01, 1000)
    assert sharpe_probabilistico(bueno, 0.0) > 0.99
    ruido = rng.normal(0.0, 0.01, 1000)
    assert 0.0 < sharpe_probabilistico(ruido, 0.0) < 1.0
    # Fórmula a mano
    sr = sharpe(bueno)
    g3, g4 = stats.skew(bueno), stats.kurtosis(bueno, fisher=False)
    z = (sr - 0.05) * math.sqrt(len(bueno) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    assert sharpe_probabilistico(bueno, 0.05) == pytest.approx(stats.norm.cdf(z))


def test_dsr_con_una_prueba_coincide_con_psr_cero():
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, 500)
    resultado = sharpe_deflactado(r, sharpes_pruebas=[sharpe(r)])
    assert resultado["n_pruebas"] == 1
    assert resultado["sr_umbral"] == 0.0
    assert resultado["dsr"] == pytest.approx(sharpe_probabilistico(r, 0.0))
    assert set(resultado) >= {"dsr", "sr", "sr_umbral", "n_pruebas", "T", "asimetria", "curtosis"}


def test_dsr_decrece_con_mas_pruebas():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0008, 0.01, 750)
    sr = sharpe(r)
    dsrs = []
    for n in (2, 10, 100, 1000):
        sharpes = np.random.default_rng(n).normal(0, 0.05, n)
        dsrs.append(sharpe_deflactado(r, sharpes_pruebas=sharpes, n_pruebas=n)["dsr"])
    assert dsrs == sorted(dsrs, reverse=True)
    assert dsrs[0] > dsrs[-1]
    assert sr > 0


def test_dsr_ruido_con_muchas_pruebas_es_bajo_y_acepta_matriz():
    rng = np.random.default_rng(5)
    matriz = pd.DataFrame(rng.normal(0, 0.01, (500, 200)))
    sharpes = sharpe(matriz)
    elegida = matriz[sharpes.idxmax()]
    resultado = sharpe_deflactado(elegida, matriz_retornos=matriz)
    assert resultado["n_pruebas"] == 200
    assert resultado["dsr"] < 0.95
    # El PSR(0) de la mejor del ruido engaña (alto); el DSR lo corrige
    assert sharpe_probabilistico(elegida, 0.0) > resultado["dsr"]


def test_dsr_exige_un_origen_de_sharpes():
    with pytest.raises(ValueError):
        sharpe_deflactado(np.zeros(10))
    with pytest.raises(ValueError):
        sharpe_deflactado(np.zeros(10), sharpes_pruebas=[0.1], matriz_retornos=pd.DataFrame(np.zeros((5, 2))))


def test_pbo_ruido_puro_es_alto():
    rng = np.random.default_rng(6)
    resultado = pbo_cscv(pd.DataFrame(rng.normal(0, 0.01, (1000, 50))), n_bloques=16)
    assert resultado["pbo"] >= 0.4
    assert resultado["n_combinaciones"] == 5000  # C(16,8)=12870 > 5000 -> muestreo
    assert len(resultado["lambdas"]) == len(resultado["pares_is_oos"]) == 5000
    assert resultado["n_configuraciones"] == 50


def test_pbo_con_columna_claramente_superior_es_bajo():
    rng = np.random.default_rng(7)
    matriz = rng.normal(0, 0.01, (1000, 30))
    matriz[:, 7] += 0.004  # media muy superior de forma persistente
    resultado = pbo_cscv(matriz, n_bloques=10)
    assert resultado["pbo"] < 0.1
    assert resultado["n_combinaciones"] == 252  # C(10,5), sin muestreo
    assert resultado["prob_perdida_oos"] < 0.1


def test_pbo_ruido_tiene_pendiente_negativa_y_semilla_reproducible():
    rng = np.random.default_rng(8)
    matriz = rng.normal(0, 0.01, (400, 40))
    a = pbo_cscv(matriz, n_bloques=16, semilla=1)
    b = pbo_cscv(matriz, n_bloques=16, semilla=1)
    assert a["pbo"] == b["pbo"]
    assert a["pendiente"] < 0  # degradación IS -> OOS


def test_pbo_validaciones():
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 3)), n_bloques=5)
    matriz = np.random.default_rng(0).normal(size=(100, 3))
    matriz[3, 1] = np.nan
    with pytest.raises(ValueError):
        pbo_cscv(matriz, n_bloques=4)


def test_prueba_rachas_ejemplo_a_mano():
    # + + - - + - : n+=3, n-=3, n=6, R=4, mu=4, sigma^2=1.2 -> z=0, p=1
    r = prueba_rachas([1, 1, -1, -1, 1, -1])
    assert r["rachas"] == 4
    assert r["mu"] == pytest.approx(4.0)
    assert r["sigma"] == pytest.approx(math.sqrt(1.2))
    assert r["z"] == pytest.approx(0.0)
    assert r["p_valor"] == pytest.approx(1.0)
    # ++++---- : R=2, mu=5, sigma^2=768/448
    r = prueba_rachas([1, 1, 1, 1, -1, -1, -1, -1])
    assert r["rachas"] == 2
    assert r["mu"] == pytest.approx(5.0)
    assert r["sigma"] == pytest.approx(math.sqrt(768 / 448))
    assert r["z"] == pytest.approx(-3 / math.sqrt(768 / 448))
    assert r["p_valor"] == pytest.approx(2 * stats.norm.cdf(r["z"]))


def test_prueba_rachas_descarta_ceros_y_casos_degenerados():
    assert prueba_rachas([1, 0, 1, -1, 0, -1])["rachas"] == 2
    assert math.isnan(prueba_rachas([1, 1, 1])["p_valor"])
    assert prueba_rachas([1, -1] * 50)["z"] > 3  # alternancia perfecta


def test_indice_ulcer_y_martin_equity_sencilla():
    # Sube a 100, cae a 90 y vuelve a 90: drawdowns en % = [0, 0, -10, -10]
    equity = [100.0, 100.0, 90.0, 90.0]
    assert indice_ulcer(equity) == pytest.approx(math.sqrt((0 + 0 + 100 + 100) / 4))
    assert indice_ulcer([1.0, 2.0, 3.0]) == 0.0
    # Martin: retornos [+10%, -10%] -> equity [1, 1.1, 0.99]
    retornos = [0.10, -0.10]
    equity = np.array([1.0, 1.1, 0.99])
    ulcer = math.sqrt(np.mean((100 * (equity / np.maximum.accumulate(equity) - 1)) ** 2))
    anual = 0.99 ** (252 / 2) - 1
    assert ratio_martin(retornos) == pytest.approx(100 * anual / ulcer)
    assert math.isnan(ratio_martin([0.01, 0.01, 0.01]))  # sin drawdown


def test_p_valor_permutacion_con_correccion_mas_uno():
    assert p_valor_permutacion(5.0, [1, 2, 3, 4]) == pytest.approx(1 / 5)  # nunca 0
    assert p_valor_permutacion(2.5, [1, 2, 3, 4]) == pytest.approx(3 / 5)
    assert p_valor_permutacion(0.0, [1, 2, 3, 4]) == pytest.approx(1.0)
    assert p_valor_permutacion(2.0, [2.0, np.nan, 1.0]) == pytest.approx(2 / 3)  # >= y sin NaN


# --------------------------------------------------------------------------- permutación


def test_permutacion_conserva_multiconjunto_primero_y_ultimo():
    datos = {"A": _ohlcv_sintetico(1), "B": _ohlcv_sintetico(2)}
    perm = permutar_precios(datos, semilla=10)
    for t in datos:
        orig, nuevo = datos[t]["Close"], perm[t]["Close"]
        assert nuevo.iloc[0] == orig.iloc[0]
        assert nuevo.iloc[-1] == pytest.approx(orig.iloc[-1], rel=1e-9)
        assert not np.allclose(nuevo.to_numpy(), orig.to_numpy())
        # Multiconjunto de retornos intrabarra + gap (cierre-cierre cambia, pero la suma total no)
        assert np.sum(_logret(nuevo)) == pytest.approx(np.sum(_logret(orig)))
        o_o, o_n = datos[t]["Open"].to_numpy()[1:], perm[t]["Open"].to_numpy()[1:]
        gaps_o = np.log(o_o) - np.log(datos[t]["Close"].to_numpy()[:-1])
        gaps_n = np.log(o_n) - np.log(perm[t]["Close"].to_numpy()[:-1])
        assert np.sort(gaps_n) == pytest.approx(np.sort(gaps_o))
        intra_o = np.log(datos[t]["Close"].to_numpy()[1:]) - np.log(o_o)
        intra_n = np.log(perm[t]["Close"].to_numpy()[1:]) - np.log(o_n)
        assert np.sort(intra_n) == pytest.approx(np.sort(intra_o))


def test_permutacion_alto_bajo_coherentes():
    perm = permutar_precios({"A": _ohlcv_sintetico(3)}, semilla=1)["A"]
    assert (perm["High"] >= perm[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (perm["Low"] <= perm[["Open", "Close"]].min(axis=1) + 1e-9).all()
    assert (perm["High"] >= perm["Low"]).all()


def test_permutacion_solo_cierre_conserva_multiconjunto_de_retornos():
    df = _ohlcv_sintetico(4)[["Close"]]
    perm = permutar_precios({"A": df}, semilla=2)["A"]
    assert perm["Close"].iloc[0] == df["Close"].iloc[0]
    assert perm["Close"].iloc[-1] == pytest.approx(df["Close"].iloc[-1])
    assert np.sort(_logret(perm["Close"])) == pytest.approx(np.sort(_logret(df["Close"])))
    assert not np.allclose(perm["Close"], df["Close"])


def test_permutacion_conserva_correlacion_entre_activos():
    rng = np.random.default_rng(0)
    n = 1500
    comun = rng.normal(0, 0.01, n)
    indice = pd.bdate_range("2015-01-01", periods=n)
    datos = {}
    for t, semilla in (("A", 1), ("B", 2)):
        r = 0.8 * comun + 0.6 * np.random.default_rng(semilla).normal(0, 0.01, n)
        datos[t] = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(r))}, index=indice)
    corr_real = np.corrcoef(_logret(datos["A"]["Close"]), _logret(datos["B"]["Close"]))[0, 1]
    perm = permutar_precios(datos, semilla=5)
    corr_perm = np.corrcoef(_logret(perm["A"]["Close"]), _logret(perm["B"]["Close"]))[0, 1]
    assert corr_real > 0.5
    assert corr_perm == pytest.approx(corr_real, abs=1e-9)  # misma permutación -> idéntica
    # La estructura temporal sí cambia
    assert not np.allclose(_logret(perm["A"]["Close"]), _logret(datos["A"]["Close"]))


def test_permutacion_respeta_inicio():
    datos = {"A": _ohlcv_sintetico(6)}
    inicio = datos["A"].index[100]
    perm = permutar_precios(datos, semilla=3, inicio=inicio)["A"]
    # Hasta la ancla (fila 100 incluida) todo real
    pd.testing.assert_frame_equal(perm.iloc[:101], datos["A"].iloc[:101])
    assert not np.allclose(perm["Close"].iloc[101:], datos["A"]["Close"].iloc[101:])
    assert perm["Close"].iloc[-1] == pytest.approx(datos["A"]["Close"].iloc[-1])
    # inicio como cadena
    perm2 = permutar_precios(datos, semilla=3, inicio=str(inicio.date()))["A"]
    pd.testing.assert_frame_equal(perm, perm2)


def test_permutacion_determinista_por_semilla():
    datos = {"A": _ohlcv_sintetico(7), "B": _ohlcv_sintetico(8)}
    a = permutar_precios(datos, semilla=42)
    b = permutar_precios(datos, semilla=42)
    c = permutar_precios(datos, semilla=43)
    for t in datos:
        pd.testing.assert_frame_equal(a[t], b[t])
    assert not np.allclose(a["A"]["Close"], c["A"]["Close"])


def test_permutacion_conserva_indice_columnas_tipos_y_no_muta_entrada():
    datos = {"A": _ohlcv_sintetico(9), "B": _ohlcv_sintetico(10, minuscula=True)}
    copia = {t: df.copy() for t, df in datos.items()}
    perm = permutar_precios(datos, semilla=1)
    for t, original in datos.items():
        pd.testing.assert_index_equal(perm[t].index, original.index)
        assert list(perm[t].columns) == list(original.columns)
        assert (perm[t].dtypes == original.dtypes).all()
        pd.testing.assert_frame_equal(original, copia[t])  # entrada intacta
    # Volume permutado: mismo multiconjunto
    assert np.sort(perm["A"]["Volume"].to_numpy()) == pytest.approx(np.sort(datos["A"]["Volume"].to_numpy()))
    assert not np.array_equal(perm["A"]["Volume"].to_numpy(), datos["A"]["Volume"].to_numpy())
    assert "close" in perm["B"].columns


@pytest.mark.parametrize("nombre_ajustado", ["Adj Close", "adjusted_close"])
def test_permutacion_columnas_ajustadas_coherentes(nombre_ajustado):
    df = _ohlcv_sintetico(11, ajustado=nombre_ajustado)
    factor_orig = df[nombre_ajustado] / df["Close"]
    perm = permutar_precios({"A": df}, semilla=4)["A"]
    # El factor por fila se conserva y el multiconjunto de retornos AJUSTADOS también
    pd.testing.assert_series_equal(perm[nombre_ajustado] / perm["Close"], factor_orig, check_names=False, atol=1e-9)
    assert np.sort(_logret(perm[nombre_ajustado])[0:]).sum() == pytest.approx(np.sum(_logret(df[nombre_ajustado])))
    assert perm[nombre_ajustado].iloc[-1] == pytest.approx(df[nombre_ajustado].iloc[-1])
    assert (perm["High"] >= perm[["Open", "Close"]].max(axis=1) - 1e-9).all()


def test_permutacion_tickers_con_fechas_distintas():
    indice_a = pd.bdate_range("2020-01-01", periods=300)
    indice_b = indice_a[50:]  # B empieza 50 días después
    datos = {"A": _ohlcv_sintetico(12, fechas=indice_a), "B": _ohlcv_sintetico(13, fechas=indice_b)}
    perm = permutar_precios(datos, semilla=2)
    # Antes de la primera fecha común, A queda real; el ancla (primera fila común) también
    pd.testing.assert_frame_equal(perm["A"].iloc[:51], datos["A"].iloc[:51])
    assert not np.allclose(perm["A"]["Close"].iloc[52:], datos["A"]["Close"].iloc[52:])
    pd.testing.assert_index_equal(perm["B"].index, datos["B"].index)
    assert perm["B"]["Close"].iloc[0] == datos["B"]["Close"].iloc[0]


def test_permutacion_bloques():
    datos = {"A": _ohlcv_sintetico(14, n=400)}
    perm = permutar_precios(datos, semilla=1, longitud_bloque=20)["A"]
    assert perm["Close"].iloc[-1] == pytest.approx(datos["A"]["Close"].iloc[-1])
    assert np.sort(_logret(perm["Close"])).sum() == pytest.approx(_logret(datos["A"]["Close"]).sum())
    # Con bloques se conserva más autocorrelación de |retorno| (clustering) que barra a barra
    rng = np.random.default_rng(0)
    n = 4000
    sigma = np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.5)  # régimen de volatilidad persistente
    sigma = 0.01 * sigma / sigma.mean()
    df = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(sigma * rng.normal(size=n)))},
                      index=pd.bdate_range("2000-01-01", periods=n))

    def autocorr_abs(serie):
        x = np.abs(_logret(serie))
        return np.corrcoef(x[:-1], x[1:])[0, 1]

    barra = permutar_precios({"A": df}, semilla=1, longitud_bloque=1)["A"]["Close"]
    bloque = permutar_precios({"A": df}, semilla=1, longitud_bloque=50)["A"]["Close"]
    assert autocorr_abs(df["Close"]) > 0.1
    assert abs(autocorr_abs(barra)) < 0.1
    assert autocorr_abs(bloque) > autocorr_abs(barra) + 0.1


def test_permutacion_errores():
    with pytest.raises(ValueError):
        permutar_precios({}, semilla=1)
    with pytest.raises(ValueError):
        permutar_precios({"A": _ohlcv_sintetico(1)[["Volume"]]}, semilla=1)
    df = _ohlcv_sintetico(1)
    df.iloc[150, df.columns.get_loc("Close")] = np.nan
    with pytest.raises(ValueError):
        permutar_precios({"A": df}, semilla=1)
    with pytest.raises(ValueError):
        permutar_precios({"A": _ohlcv_sintetico(1)}, semilla=1, longitud_bloque=0)


def test_permutacion_inicio_cerca_del_final_devuelve_copia():
    df = _ohlcv_sintetico(15, n=50)
    perm = permutar_precios({"A": df}, semilla=1, inicio=df.index[-2])["A"]
    pd.testing.assert_frame_equal(perm, df)


# --------------------------------------------------------------------------- walk-forward


def _matriz_ruido(filas=600, columnas=5, semilla=0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    return pd.DataFrame(
        rng.normal(0, 0.01, (filas, columnas)),
        index=pd.bdate_range("2018-01-01", periods=filas),
        columns=[f"c{i}" for i in range(columnas)],
    )


def test_walk_forward_longitudes_y_estructura():
    matriz = _matriz_ruido(filas=600)
    serie, elecciones, valor = seleccion_walk_forward(matriz, ventana_entreno=100, paso=50)
    assert len(serie) == 500
    assert serie.index[0] == matriz.index[100] and serie.index[-1] == matriz.index[-1]
    assert len(elecciones) == 10  # t = 100, 150, ..., 550
    assert list(elecciones.index) == list(matriz.index[100::50])
    assert set(elecciones["columna"]) <= set(matriz.columns)
    assert valor == pytest.approx(sharpe(serie.to_numpy()))
    # Cada tramo usa los retornos de la columna elegida
    t0 = elecciones.index[0]
    columna0 = elecciones["columna"].iloc[0]
    pd.testing.assert_series_equal(serie.loc[t0:].iloc[:50], matriz[columna0].loc[t0:].iloc[:50], check_names=False)


def test_walk_forward_ultimo_tramo_corto():
    matriz = _matriz_ruido(filas=230)
    serie, elecciones, _ = seleccion_walk_forward(matriz, ventana_entreno=100, paso=60)
    assert len(serie) == 130
    assert len(elecciones) == 3  # 100, 160, 220 (tramo final de 10 filas)


def test_walk_forward_sin_look_ahead_columna_buena_solo_en_el_futuro():
    matriz = _matriz_ruido(filas=600, columnas=4, semilla=1)
    punto_cambio = 300
    matriz["futuro"] = np.random.default_rng(5).normal(0, 0.01, 600)
    matriz.loc[matriz.index[:punto_cambio], "futuro"] -= 0.01  # mala antes
    matriz.loc[matriz.index[punto_cambio:], "futuro"] += 0.05  # buenísima solo desde la fila 300
    _, elecciones, _ = seleccion_walk_forward(matriz, ventana_entreno=100, paso=20)
    antes = elecciones[elecciones.index <= matriz.index[punto_cambio]]
    assert (antes["columna"] != "futuro").all()
    despues = elecciones[elecciones.index > matriz.index[punto_cambio + 20]]
    assert (despues["columna"] == "futuro").any()  # y sí se elige cuando ya la ha visto


def test_walk_forward_la_eleccion_no_depende_de_datos_futuros():
    matriz = _matriz_ruido(filas=400, columnas=6, semilla=2)
    _, base, _ = seleccion_walk_forward(matriz, ventana_entreno=80, paso=40)
    alterada = matriz.copy()
    alterada.iloc[250:] = np.random.default_rng(99).normal(0.05, 0.02, alterada.iloc[250:].shape)
    _, nueva, _ = seleccion_walk_forward(alterada, ventana_entreno=80, paso=40)
    previas = base.index[base.index <= matriz.index[250]]
    pd.testing.assert_frame_equal(base.loc[previas], nueva.loc[previas])


def test_walk_forward_nan_y_minimo_observaciones():
    matriz = _matriz_ruido(filas=300, columnas=3, semilla=3)
    matriz.iloc[:150, 2] = np.nan  # c2 no existe hasta la fila 150
    _, elecciones, _ = seleccion_walk_forward(matriz, ventana_entreno=100, paso=50)
    assert elecciones["columna"].iloc[0] != "c2"  # ventana completa exigida
    assert elecciones["columna"].iloc[1] != "c2"  # t=150: ventana [50,150) aún con NaN
    # Si ninguna columna es elegible, el tramo queda en NaN
    todo_nan = matriz.copy()
    todo_nan.iloc[:, :] = np.nan
    serie, eleccion, _ = seleccion_walk_forward(todo_nan, ventana_entreno=100, paso=50)
    assert serie.isna().all() and eleccion["columna"].isna().all()


def test_walk_forward_metrica_personalizada_y_validaciones():
    matriz = _matriz_ruido(filas=300, columnas=3, semilla=4)
    _, elecciones, _ = seleccion_walk_forward(matriz, 100, 50, metrica=lambda x: -np.nanmean(x, axis=0))
    # La peor media histórica gana con la métrica invertida
    t = matriz.index.get_loc(elecciones.index[0])
    peor = matriz.iloc[t - 100 : t].mean().idxmin()
    assert elecciones["columna"].iloc[0] == peor
    with pytest.raises(ValueError):
        seleccion_walk_forward(matriz, ventana_entreno=0, paso=10)
    with pytest.raises(ValueError):
        seleccion_walk_forward(matriz, ventana_entreno=300, paso=10)
