"""
Tests de las estrategias de señales de ejemplo y su integración (fase 2 del plan): `cruce_medias`,
`canal_rsi_volatilidad` y el puente `senales_tecnicas` con el motor, el registro, los barridos, los marcos de
comparación y `verificar_sin_lookahead`. Datos sintéticos, sin red.

Responsabilidad: comprobar que las estrategias plantilla corren con `ejecutar_con_datos`, aparecen en el
registro con su formulario, no miran al futuro y se lanzan en un marco como las demás.

Se relaciona con: tests/test_senales.py (máquina de estados), estrategias_usuario/cruce_medias.py,
estrategias_usuario/canal_rsi_volatilidad.py, src/ypyquant/estrategias/senales_tecnicas.py, causalidad.py y marcos.py.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias import registro
from ypyquant.estrategias.causalidad import verificar_sin_lookahead
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, cargar_datos_experimento, ejecutar_con_datos
from ypyquant.estrategias.marcos import MarcoComparacion, aplicar_marco
from ypyquant.estrategias.senales import TICKER_EFECTIVO, EstrategiaSenales

# ---- 2.4 integración con el motor y el registro -------------------------------------------------------


UNIVERSO = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _config(estrategia, **cambios) -> ConfiguracionExperimento:
    base = {"estrategia": estrategia, "universo": UNIVERSO, "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}
    return ConfiguracionExperimento(**{**base, **cambios})


def test_cruce_medias_corre_con_el_motor_y_devuelve_metricas_y_pesos_con_efectivo():
    config = _config("cruce_medias", parametros={"rapida": 10, "lenta": 40, "stop_perdida": 0.1})
    resultado = ejecutar_con_datos(config, cargar_datos_experimento(config, FuenteSintetica()))
    assert resultado.metricas["costes_sobre_ganancia_bruta"] is not None
    assert resultado.metricas["n_rebalanceos"] > 100
    assert not resultado.equity.empty and resultado.equity["equity"].iloc[-1] > 0
    assert set(resultado.pesos["ticker"]) <= set(UNIVERSO) | {TICKER_EFECTIVO}
    assert "motivo" in resultado.operaciones.columns and resultado.operaciones["motivo"].notna().all()
    assert set(resultado.operaciones["motivo"]) <= {"entrada_larga", "senal_salida", "stop_loss", "ajuste", "trailing",
                                                    "take_profit", "tiempo", "salida_universo"}


def test_las_estrategias_sin_senales_no_ganan_columna_motivo():
    config = _config("pesos_iguales")
    resultado = ejecutar_con_datos(config, cargar_datos_experimento(config, FuenteSintetica()))
    assert "motivo" not in resultado.operaciones.columns


def test_una_rejilla_rapida_por_lenta_corre_con_celdas_omitidas():
    from ypyquant.estrategias.experimentos import expandir_rejilla_con_omitidas

    configs, omitidas = expandir_rejilla_con_omitidas(
        _config("cruce_medias"), {"rapida": [10, 50], "lenta": [30, 60], "stop_perdida": [None, 0.1]})
    assert len(configs) == 6 and len(omitidas) == 2      # (50, 30) viola rapida < lenta, con y sin stop
    datos = cargar_datos_experimento(configs[0], FuenteSintetica())
    resultados = [ejecutar_con_datos(c, datos, completo=False) for c in configs]
    assert all(r.metricas["n_rebalanceos"] > 100 for r in resultados)


def test_las_estrategias_de_ejemplo_aparecen_en_el_registro_con_formulario():
    nombres = {e["nombre"]: e for e in registro.listar_estrategias()}
    for nombre in ("cruce_medias", "canal_rsi_volatilidad", "senales_tecnicas"):
        propiedades = nombres[nombre]["parametros_schema"]["properties"]
        assert {"max_posiciones", "stop_perdida", "stop_trailing", "enfriamiento_dias"} <= set(propiedades)
        assert nombres[nombre]["frecuencia_rebalanceo"] == "daily"
    # (`origen` depende del orden de los tests si otro recarga el registro; el módulo identifica el archivo de usuario)
    assert registro.obtener_estrategia("cruce_medias").__module__ == "estrategia_usuario_cruce_medias"
    assert nombres["senales_tecnicas"]["origen"] == "incluida"
    assert {"rapida", "lenta"} <= set(nombres["cruce_medias"]["parametros_schema"]["properties"])


def test_cruce_medias_rechaza_rapida_mayor_o_igual_que_lenta():
    with pytest.raises(ValidationError, match="rapida debe ser menor que lenta"):
        registro.obtener_estrategia("cruce_medias").Parametros(rapida=50, lenta=20)


# ---- 2.5 causalidad ---------------------------------------------------------------------------------------


def _datos_aleatorios(tickers=("AAA", "BBB", "CCC"), n=260, semilla=7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2021-01-04", periods=n)
    datos = {}
    for t in tickers:
        cierre = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
        datos[t] = pd.DataFrame({"Close": cierre, "adjusted_close": cierre,
                                 "Volume": rng.integers(1e6, 5e6, n).astype(float)}, index=fechas)
    return datos


FECHAS_MUESTRA = ["2021-08-02", "2021-10-15", "2021-12-01", "2022-01-14"]


@pytest.mark.parametrize("nombre, parametros", [
    ("cruce_medias", {"rapida": 5, "lenta": 20, "stop_perdida": 0.1, "stop_trailing": 0.1, "max_posiciones": 2}),
    ("canal_rsi_volatilidad", {"ventana_canal": 20, "ventana_salida": 8, "ventana_rsi": 7, "max_posiciones": 2}),
    ("senales_tecnicas", {"max_posiciones": 2, "umbral": 0.2}),
])
def test_las_estrategias_de_senales_incluidas_no_miran_al_futuro(nombre, parametros):
    clase = registro.obtener_estrategia(nombre)
    assert verificar_sin_lookahead(clase, parametros, _datos_aleatorios(), FECHAS_MUESTRA) == []


def test_una_subclase_tramposa_con_shift_negativo_en_la_entrada_es_detectada():
    class Tramposa(EstrategiaSenales):
        nombre = "tramposa_prueba"

        def senal_entrada(self, cierres, indicadores):
            return cierres.shift(-1) > cierres            # mira el día siguiente

        def senal_salida(self, cierres, indicadores):
            return cierres.shift(-1) < cierres

    errores = verificar_sin_lookahead(Tramposa, {"max_posiciones": 3, "peso_maximo": 0.5}, _datos_aleatorios(), FECHAS_MUESTRA)
    assert errores and all("look-ahead" in e for e in errores)


# ---- 2.6 puente con las señales heredadas ---------------------------------------------------------------


def test_senales_tecnicas_devuelve_pesos_validos_con_el_generador_tecnico_por_defecto():
    datos = _datos_aleatorios(tickers=("AAA", "BBB", "CCC", "DDD"), n=300)
    clase = registro.obtener_estrategia("senales_tecnicas")
    estrategia = clase({"max_posiciones": 4, "umbral": 0.1})
    funcion = estrategia.como_funcion()
    fechas = list(datos["AAA"].index[-60:])
    pesos = [funcion(f.to_pydatetime(), datos) for f in fechas]
    assert all(sum(p.values()) == pytest.approx(1.0) for p in pesos)
    assert all(0 <= w <= 1 for p in pesos for w in p.values())
    assert any(set(p) - {TICKER_EFECTIVO} for p in pesos)       # alguna posición se abre
    # Diferencia documentada: el tamaño de entrada es fijo (1/max_posiciones); luego solo deriva con el precio
    assert all(w <= 0.40 for p in pesos for t, w in p.items() if t != TICKER_EFECTIVO)
    entradas = [e for e in estrategia.eventos if e['tipo'] == 'entrada']
    assert entradas and all(e['motivo'] == 'entrada_larga' for e in entradas)


# ---- marcos de comparación -----------------------------------------------------------------------------


def test_una_estrategia_de_senales_se_lanza_en_un_marco_como_las_demas():
    marco = MarcoComparacion(nombre="Prueba", universo=UNIVERSO, fecha_inicio="2020-01-01", fecha_fin="2021-12-31",
                             frecuencia_rebalanceo="weekly", parametros_comunes={"estimador_sigma": "muestral"})
    config, informe = aplicar_marco(marco, "cruce_medias", {"rapida": 10, "lenta": 40})
    assert config.frecuencia_rebalanceo == "weekly"
    assert informe.no_aplicados == ["estimador_sigma"]          # no lo declara: se informa, no se rompe
    resultado = ejecutar_con_datos(config, cargar_datos_experimento(config, FuenteSintetica()))
    assert resultado.metricas["n_rebalanceos"] > 50


def test_marco_sin_frecuencia_avisa_de_que_cruce_medias_usa_la_diaria():
    marco = MarcoComparacion(nombre="Prueba", universo=UNIVERSO, fecha_inicio="2020-01-01", fecha_fin="2021-12-31",
                             frecuencia_rebalanceo=None)
    config, informe = aplicar_marco(marco, "cruce_medias")
    assert config.frecuencia_rebalanceo == "daily"
    assert any("daily" in aviso for aviso in informe.avisos)


# ---- almacén: operaciones con y sin motivo ----------------------------------------------------------------------


def test_el_almacen_mezcla_operaciones_con_motivo_y_sin_el_en_las_consultas_sql(tmp_path):
    from ypyquant.estrategias.almacen import AlmacenExperimentos

    almacen = AlmacenExperimentos(tmp_path / "experimentos")
    ids = []
    for estrategia, parametros in (("cruce_medias", {"rapida": 10, "lenta": 40}), ("pesos_iguales", {})):
        config = _config(estrategia, parametros=parametros)
        id_experimento = almacen.registrar(config)
        almacen.guardar(id_experimento, ejecutar_con_datos(config, cargar_datos_experimento(config, FuenteSintetica())))
        ids.append(id_experimento)
    tabla = almacen.ejecutar_sql("SELECT id, motivo, COUNT(*) AS n FROM operaciones GROUP BY id, motivo")
    assert set(tabla["id"]) == set(ids)                                      # ambos experimentos aparecen
    assert tabla["motivo"].notna().any() and tabla["motivo"].isna().any()   # los sin señales llevan motivo nulo


# ---- modo largo-corto de las estrategias migradas ----------------------------------------------------------


@pytest.mark.parametrize("nombre, parametros", [
    ("cruce_medias", {"rapida": 5, "lenta": 20, "stop_perdida": 0.1, "stop_trailing": 0.1, "max_posiciones": 2}),
    ("canal_rsi_volatilidad", {"ventana_canal": 20, "ventana_salida": 8, "ventana_rsi": 7, "max_posiciones": 2}),
    ("senales_tecnicas", {"max_posiciones": 2, "umbral": 0.2}),
])
def test_el_modo_largo_corto_de_las_estrategias_incluidas_tampoco_mira_al_futuro(nombre, parametros):
    clase = registro.obtener_estrategia(nombre)
    parametros = {**parametros, "lados": "largo_corto", "exposicion_bruta_max": 1.5, "permitir_giro": True}
    assert verificar_sin_lookahead(clase, parametros, _datos_aleatorios(), FECHAS_MUESTRA) == []


@pytest.mark.parametrize("nombre, parametros", [
    ("cruce_medias", {"rapida": 10, "lenta": 40}),
    ("canal_rsi_volatilidad", {"ventana_canal": 20, "ventana_salida": 8, "ventana_rsi": 7}),
    ("senales_tecnicas", {"max_posiciones": 3, "umbral": 0.2}),
])
def test_las_estrategias_migradas_abren_cortos_con_lados_largo_corto_y_el_modo_cambia_la_huella(nombre, parametros):
    base = _config(nombre, parametros=parametros)
    largo_corto = _config(nombre, parametros={**parametros, "lados": "largo_corto", "exposicion_bruta_max": 1.5})
    assert base.normalizada().identificador() != largo_corto.normalizada().identificador()
    assert base.normalizada().identificador() == _config(nombre, parametros={**parametros, "lados": "largo"}).normalizada().identificador()
    resultado = ejecutar_con_datos(largo_corto, cargar_datos_experimento(largo_corto, FuenteSintetica()))
    assert (resultado.pesos["peso"] < 0).any()
    assert "entrada_corta" in set(resultado.operaciones["motivo"])
    assert resultado.metricas["exposicion_bruta_maxima"] < 1.8   # el límite de 1,5 rige al entrar; la deriva puede superarlo algo
