"""
Tests de `EstrategiaSenales` (fase 2 del plan): máquina de estados, pesos con CASH, idempotencia,
integración con el motor, causalidad y estrategias de ejemplo. Datos sintéticos, sin red.

Responsabilidad: fijar el contrato de ypyquant.estrategias.senales con precios construidos a mano
(series que suben, cruzan y bajan) para que cada regla (entrada, salida, stops, enfriamiento) se
compruebe en un día concreto, y comprobar que las estrategias de ejemplo (`cruce_medias`,
`canal_rsi_volatilidad`, `senales_tecnicas`) no miran al futuro y corren con el motor.

Se relaciona con: src/ypyquant/estrategias/senales.py, senales_tecnicas.py, causalidad.py, experimentos.py,
marcos.py, estrategias_usuario/cruce_medias.py y estrategias_usuario/canal_rsi_volatilidad.py.
"""

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ypyquant.estrategias.base import recortar_hasta
from ypyquant.estrategias.senales import TICKER_EFECTIVO, EstrategiaSenales, ParametrosSenales
from ypyquant.portfolio.backtest_engine import BacktestConfig, BacktestEngine

N = 30
FECHAS = pd.bdate_range("2022-01-03", periods=N)


def _datos(**series) -> dict[str, pd.DataFrame]:
    """`{ticker: OHLCV}` con una serie de cierres por ticker (se rellena con el último valor hasta N)."""
    datos = {}
    for ticker, valores in series.items():
        valores = list(valores) + [valores[-1]] * (N - len(valores))
        datos[ticker] = pd.DataFrame({"Close": valores, "adjusted_close": valores}, index=FECHAS, dtype=float)
    return datos


def _tabla(tickers, **tramos) -> pd.DataFrame:
    """Tabla booleana fecha x ticker: `ticker=(desde, hasta)` es True en las posiciones [desde, hasta)."""
    tabla = pd.DataFrame(False, index=FECHAS, columns=list(tickers))
    for ticker, (desde, hasta) in tramos.items():
        tabla.iloc[desde:hasta, tabla.columns.get_loc(ticker)] = True
    return tabla


def _estrategia(entrada, salida=None, puntuacion=None, **parametros) -> EstrategiaSenales:
    """Estrategia de prueba cuyas señales son tablas fijas (sin registrar)."""

    class Manual(EstrategiaSenales):
        nombre = "manual_prueba"

        def senal_entrada(self, cierres, indicadores):
            return entrada.reindex_like(cierres).fillna(False)

        def senal_salida(self, cierres, indicadores):
            vacia = pd.DataFrame(False, index=cierres.index, columns=cierres.columns)
            return vacia if salida is None else salida.reindex_like(cierres).fillna(False)

        def puntuacion(self, cierres, indicadores):
            return None if puntuacion is None else puntuacion.reindex_like(cierres)

    parametros = {"peso_maximo": 1.0, **parametros}
    return Manual(parametros)


def _recorrer(estrategia, datos, desde=0, hasta=N) -> list[dict[str, float]]:
    """Pesos día a día como lo haría el motor (`como_funcion`: recorte, preparar y estado)."""
    funcion = estrategia.como_funcion()
    return [funcion(FECHAS[i].to_pydatetime(), datos) for i in range(desde, hasta)]


# ---- 2.1 máquina de estados ---------------------------------------------------------------------------


def test_entra_el_dia_de_la_senal_con_peso_1_entre_max_posiciones_y_el_resto_en_efectivo():
    datos = _datos(AAA=[100] * N)
    pesos = _recorrer(_estrategia(_tabla(["AAA"], AAA=(5, N)), max_posiciones=2), datos)
    assert all(p == {TICKER_EFECTIVO: 1.0} for p in pesos[:5])
    assert pesos[5] == pytest.approx({"AAA": 0.5, TICKER_EFECTIVO: 0.5})


def test_con_un_hueco_y_dos_candidatos_entra_el_de_mayor_puntuacion_y_el_empate_va_alfabetico():
    datos = _datos(AAA=[100] * N, BBB=[100] * N)
    entrada = _tabla(["AAA", "BBB"], AAA=(5, N), BBB=(5, N))
    puntuacion = pd.DataFrame({"AAA": 1.0, "BBB": 2.0}, index=FECHAS)
    pesos = _recorrer(_estrategia(entrada, puntuacion=puntuacion, max_posiciones=1), datos)
    assert pesos[5] == {"BBB": 1.0}
    empate = _recorrer(_estrategia(entrada, max_posiciones=1), datos)  # sin puntuación: orden alfabético
    assert empate[5] == {"AAA": 1.0}


def test_no_entra_si_la_entrada_y_la_salida_coinciden_ni_en_un_ticker_sin_precio_ese_dia():
    datos = _datos(AAA=[100] * N, BBB=[100] * 5 + [np.nan] + [100] * (N - 6))
    entrada = _tabla(["AAA", "BBB"], AAA=(5, N), BBB=(5, N))
    salida = _tabla(["AAA", "BBB"], AAA=(5, 6))
    pesos = _recorrer(_estrategia(entrada, salida, max_posiciones=2), datos)
    assert pesos[5] == {TICKER_EFECTIVO: 1.0}
    assert pesos[6] == {"AAA": 0.5, "BBB": 0.5}


def test_sale_por_la_senal_de_salida_y_no_reentra_mientras_la_salida_siga_activa():
    datos = _datos(AAA=[100] * N)
    estrategia = _estrategia(_tabla(["AAA"], AAA=(5, N)), _tabla(["AAA"], AAA=(10, N)), max_posiciones=1)
    pesos = _recorrer(estrategia, datos)
    assert pesos[9] == {"AAA": 1.0}
    assert pesos[10] == {TICKER_EFECTIVO: 1.0} and pesos[11] == {TICKER_EFECTIVO: 1.0}
    salida = [e for e in estrategia.eventos if e["tipo"] == "salida"]
    assert [(e["fecha"], e["motivo"]) for e in salida] == [(f"{FECHAS[10]:%Y-%m-%d}", "senal_salida")]


def test_stop_perdida_dispara_con_caida_del_9_por_ciento_y_el_enfriamiento_impide_reentrar():
    datos = _datos(AAA=[100] * 10 + [91] * 20)
    estrategia = _estrategia(_tabla(["AAA"], AAA=(5, N)), max_posiciones=1, stop_perdida=0.08, enfriamiento_dias=3)
    pesos = _recorrer(estrategia, datos)
    assert pesos[9] == {"AAA": 1.0}
    assert pesos[10] == {TICKER_EFECTIVO: 1.0}          # 91/100 - 1 = -9 % <= -8 %
    assert pesos[11] == {TICKER_EFECTIVO: 1.0} and pesos[12] == {TICKER_EFECTIVO: 1.0}
    assert pesos[13] == {"AAA": 1.0}                    # 3 días hábiles después puede reentrar
    assert [e["motivo"] for e in estrategia.eventos if e["tipo"] == "salida"] == ["stop_loss"]


def test_stop_perdida_no_dispara_con_una_caida_menor():
    datos = _datos(AAA=[100] * 10 + [93] * 20)
    pesos = _recorrer(_estrategia(_tabla(["AAA"], AAA=(5, N)), max_posiciones=1, stop_perdida=0.08), datos)
    assert pesos[10] == {"AAA": 1.0}


def test_stop_trailing_usa_el_maximo_desde_la_entrada():
    # Entra a 100, sube a 120 y cae a 112 (-6,7 % desde el máximo: no sale) y a 110 (-8,3 %: sale)
    datos = _datos(AAA=[100] * 6 + [105, 110, 120, 112, 110])
    estrategia = _estrategia(_tabla(["AAA"], AAA=(5, N)), max_posiciones=1, stop_trailing=0.08, enfriamiento_dias=0)
    pesos = _recorrer(estrategia, datos, hasta=11)
    assert pesos[9] == {"AAA": 1.0}
    assert pesos[10] == {TICKER_EFECTIVO: 1.0}
    assert [e["motivo"] for e in estrategia.eventos if e["tipo"] == "salida"] == ["trailing"]


def test_stop_trailing_mira_el_maximo_de_los_dias_en_que_no_hubo_llamada():
    """Con rebalanceo no diario (llamadas en los días 5 y 10) el máximo de los días intermedios cuenta."""
    datos = _datos(AAA=[100] * 6 + [105, 110, 120, 112, 110])
    funcion = _estrategia(_tabla(["AAA"], AAA=(5, N)), max_posiciones=1, stop_trailing=0.08).como_funcion()
    assert funcion(FECHAS[5].to_pydatetime(), datos) == {"AAA": 1.0}
    assert funcion(FECHAS[10].to_pydatetime(), datos) == {TICKER_EFECTIVO: 1.0}


def test_take_profit_vende_al_alcanzar_la_ganancia():
    datos = _datos(AAA=[100] * 8 + [110, 116])
    estrategia = _estrategia(_tabla(["AAA"], AAA=(5, N)), max_posiciones=1, take_profit=0.15)
    pesos = _recorrer(estrategia, datos, hasta=10)
    assert pesos[8] == {"AAA": 1.0} and pesos[9] == {TICKER_EFECTIVO: 1.0}
    assert [e["motivo"] for e in estrategia.eventos if e["tipo"] == "salida"] == ["take_profit"]


def test_max_dias_posicion_sale_tras_n_sesiones_y_un_stop_no_activa_enfriamiento_si_no_es_de_perdida():
    datos = _datos(AAA=[100] * N)
    estrategia = _estrategia(_tabla(["AAA"], AAA=(5, N)), max_posiciones=1, max_dias_posicion=3, enfriamiento_dias=5)
    pesos = _recorrer(estrategia, datos, hasta=10)
    assert pesos[7] == {"AAA": 1.0} and pesos[8] == {TICKER_EFECTIVO: 1.0}
    assert pesos[9] == {"AAA": 1.0}   # la salida por tiempo no bloquea la reentrada (el día siguiente ya entra)


def test_un_ticker_que_sale_del_universo_se_vende_con_motivo_fuera_universo():
    datos = _datos(AAA=[100] * N, BBB=[100] * N)
    estrategia = _estrategia(_tabla(["AAA", "BBB"], AAA=(5, N), BBB=(5, N)), max_posiciones=2)
    funcion = estrategia.como_funcion(filtro_universo=lambda fecha, d: {t: v for t, v in d.items() if t == "AAA" or fecha < FECHAS[8]})
    pesos = [funcion(FECHAS[i].to_pydatetime(), datos) for i in range(5, 10)]
    assert set(pesos[2]) == {"AAA", "BBB"} and set(pesos[3]) == {"AAA", TICKER_EFECTIVO}
    assert [e["motivo"] for e in estrategia.eventos if e["tipo"] == "salida"] == ["salida_universo"]


def test_salida_personalizada_recibe_la_posicion_y_los_cierres_hasta_la_fecha():
    llamadas = []

    class ConRegla(EstrategiaSenales):
        nombre = "con_regla_prueba"

        def senal_entrada(self, cierres, indicadores):
            return pd.DataFrame(True, index=cierres.index, columns=cierres.columns)

        def senal_salida(self, cierres, indicadores):
            return pd.DataFrame(False, index=cierres.index, columns=cierres.columns)

        def salida_personalizada(self, ticker, fecha, posicion, cierres_hasta):
            llamadas.append((ticker, fecha, len(cierres_hasta)))
            return len(cierres_hasta) >= 8

    pesos = _recorrer(ConRegla({"max_posiciones": 1, "peso_maximo": 1.0, "enfriamiento_dias": 0}), _datos(AAA=[100] * N), hasta=9)
    assert pesos[6] == {"AAA": 1.0} and pesos[7] == {TICKER_EFECTIVO: 1.0}
    assert llamadas[0][0] == "AAA" and llamadas[0][2] == 2  # primera evaluación el día 1 (entra el 0)


def test_ponderacion_inversa_volatilidad_da_mas_peso_al_activo_menos_volatil():
    rng = np.random.default_rng(1)
    calma = list(100 * np.cumprod(1 + rng.normal(0, 0.002, N)))
    nerviosa = list(100 * np.cumprod(1 + rng.normal(0, 0.03, N)))
    datos = _datos(AAA=calma, BBB=nerviosa)
    estrategia = _estrategia(_tabla(["AAA", "BBB"], AAA=(25, N), BBB=(25, N)), max_posiciones=2,
                             ponderacion="inversa_volatilidad", ventana_volatilidad=10)
    pesos = _recorrer(estrategia, datos, hasta=26)[25]
    assert pesos["AAA"] > pesos["BBB"] > 0
    assert sum(pesos.values()) == pytest.approx(1.0)
    assert pesos["AAA"] + pesos["BBB"] == pytest.approx(1.0)  # el presupuesto total de 2 huecos de 2 es todo el capital


def test_peso_maximo_recorta_la_entrada_y_deja_el_resto_en_efectivo():
    pesos = _recorrer(_estrategia(_tabla(["AAA"], AAA=(5, N)), max_posiciones=2, peso_maximo=0.3), _datos(AAA=[100] * N))
    assert pesos[5] == pytest.approx({"AAA": 0.3, TICKER_EFECTIVO: 0.7})


def test_parametros_validan_limites_y_rechazan_campos_desconocidos():
    with pytest.raises(ValidationError):
        ParametrosSenales(max_posiciones=0)
    with pytest.raises(ValidationError):
        ParametrosSenales(stop_perdida=1.5)
    with pytest.raises(ValidationError):
        ParametrosSenales(inventado=1)


def test_las_tablas_de_senales_deben_ser_booleanas_con_la_forma_de_los_cierres():
    class Mala(EstrategiaSenales):
        nombre = "mala_prueba"

        def senal_entrada(self, cierres, indicadores):
            return cierres.iloc[:5]            # forma incorrecta

        def senal_salida(self, cierres, indicadores):
            return cierres > 0

    funcion = Mala().como_funcion()
    with pytest.raises(ValueError, match="senal_entrada"):
        funcion(FECHAS[3].to_pydatetime(), _datos(AAA=[100] * N))


# ---- 2.2 pesos ------------------------------------------------------------------------------------------


def test_pesos_suman_1_con_efectivo_y_sin_senales_todo_es_efectivo():
    datos = _datos(AAA=[100 + i for i in range(N)], BBB=[100] * N)
    con = _recorrer(_estrategia(_tabla(["AAA", "BBB"], AAA=(3, 20), BBB=(8, N)), max_posiciones=4), datos)
    assert all(sum(p.values()) == pytest.approx(1.0) for p in con)
    sin = _recorrer(_estrategia(_tabla(["AAA", "BBB"]), max_posiciones=4), datos)
    assert all(p == {TICKER_EFECTIVO: 1.0} for p in sin)


def test_la_deriva_actualiza_los_pesos_sin_cambios_y_sin_ella_se_mantienen():
    datos = _datos(AAA=[100 * 1.01 ** i for i in range(N)], BBB=[100] * N)
    entrada = _tabla(["AAA", "BBB"], AAA=(5, N), BBB=(5, N))
    con = _recorrer(_estrategia(entrada, max_posiciones=4, peso_maximo=0.25), datos)
    # Día 5: 0.25 + 0.25 + 0.5 de efectivo; día 6: AAA sube 1 %
    assert con[5]["AAA"] == pytest.approx(0.25)
    esperado = 0.25 * 1.01 / (0.25 * 1.01 + 0.25 + 0.5)
    assert con[6]["AAA"] == pytest.approx(esperado)
    assert con[6]["AAA"] > con[5]["AAA"]
    sin = _recorrer(_estrategia(entrada, max_posiciones=4, peso_maximo=0.25, mantener_deriva=False), datos)
    assert sin[6]["AAA"] == pytest.approx(0.25)


class _CargadorPlano:
    """Cargador para el motor: precios en minúscula como GrowSave; AAA crece y BBB oscila."""

    def __call__(self, ticker, inicio, fin):
        fechas = pd.bdate_range("2020-01-01", "2023-12-29")
        base = 100 * (1.0004 ** np.arange(len(fechas))) if ticker == "AAA" else 100 + 5 * np.sin(np.arange(len(fechas)) / 20)
        df = pd.DataFrame({"close": base, "adjusted_close": base}, index=fechas)
        return df[(df.index >= pd.Timestamp(inicio)) & (df.index <= pd.Timestamp(fin))]


def test_con_la_deriva_el_motor_apenas_opera_los_dias_sin_cambios():
    """Dos activos con entrada permanente: tras entrar, el turnover diario es mínimo (solo el redondeo de costes)."""
    fechas = pd.bdate_range("2020-01-01", "2023-12-29")
    entrada = pd.DataFrame(False, index=fechas, columns=["AAA", "BBB"])
    entrada.loc["2022-01-03":, :] = True
    estrategia = _estrategia(entrada, max_posiciones=2, peso_maximo=0.5)
    motor = BacktestEngine(BacktestConfig(start_date="2022-01-03", end_date="2022-12-30", initial_capital=100_000,
                                          rebalance_frequency="daily", commission_rate=0.001, slippage_pct=0.001,
                                          user_currency="USD"), cargador_precios=_CargadorPlano())
    motor.run_backtest(estrategia.como_funcion(), ["AAA", "BBB"])
    rotaciones = [r for _, r in motor.rotaciones[1:]]
    assert len(rotaciones) > 200
    assert max(rotaciones) < 0.01 and np.mean(rotaciones) < 0.002
    assert all(t.ticker != TICKER_EFECTIVO for t in motor.trades)


# ---- 2.3 idempotencia y serialización ------------------------------------------------------------------


def _estrategia_con_stop():
    datos = _datos(AAA=[100] * 10 + [91] * 20, BBB=[100 + i for i in range(N)])
    entrada = _tabla(["AAA", "BBB"], AAA=(5, N), BBB=(12, N))
    return datos, lambda: _estrategia(entrada, max_posiciones=2, peso_maximo=0.5, stop_perdida=0.08, enfriamiento_dias=3)


def test_llamar_dos_veces_con_la_misma_fecha_no_cambia_el_estado_ni_los_pesos():
    datos, fabrica = _estrategia_con_stop()
    estrategia = fabrica()
    estado = {}
    for i in range(12):
        estrategia.generar_pesos(FECHAS[i].to_pydatetime(), recortar_hasta(datos, FECHAS[i]), estado)
    copia = json.dumps(estado, sort_keys=True)
    eventos = list(estrategia.eventos)
    repetido = estrategia.generar_pesos(FECHAS[11].to_pydatetime(), recortar_hasta(datos, FECHAS[11]), estado)
    assert repetido == estado["pesos"]
    assert json.dumps(estado, sort_keys=True) == copia and estrategia.eventos == eventos


def test_el_estado_es_json_y_una_instancia_nueva_continua_igual_que_la_original():
    datos, fabrica = _estrategia_con_stop()
    original, estado = fabrica(), {}
    pesos_original = []
    for i in range(N):
        pesos_original.append(original.generar_pesos(FECHAS[i].to_pydatetime(), recortar_hasta(datos, FECHAS[i]), estado))
        if i == 9:
            guardado = json.loads(json.dumps(estado))      # a disco y de vuelta
    reanudada, estado_nuevo = fabrica(), guardado
    pesos_reanudada = [
        reanudada.generar_pesos(FECHAS[i].to_pydatetime(), recortar_hasta(datos, FECHAS[i]), estado_nuevo)
        for i in range(10, N)
    ]
    assert pesos_reanudada == pesos_original[10:]
    assert estado_nuevo == json.loads(json.dumps(estado))
