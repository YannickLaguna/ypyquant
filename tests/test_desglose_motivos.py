"""
Tests del desglose por motivo de salida (`desglose_por_motivo`) y de su aparición en las métricas y en la ficha del experimento.

Responsabilidad: comprobar con operaciones construidas a mano que cada salida se agrupa por motivo y lado de la posición con su
número, PnL, tasa de acierto y duración media (de la entrada a la salida), que los ajustes parciales no cuentan como salidas,
que las operaciones guardadas con el vocabulario anterior (`senal`, `stop_perdida`) se leen igual y que solo las estrategias de
señales (con columna `motivo`) lo traen.

Se relaciona con: ypyquant.estrategias.evaluacion (`desglose_por_motivo`, `evaluar`), almacen.py (`leer`) y
app/web/src/paginas/ExperimentoDetalle.tsx (tabla del desglose).
"""

import numpy as np
import pandas as pd
import pytest

from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.evaluacion import desglose_por_motivo
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, cargar_datos_experimento, ejecutar_con_datos

COLUMNAS = ["fecha", "ticker", "lado", "cantidad", "precio", "comision", "pnl", "motivo"]


def _operaciones(filas) -> pd.DataFrame:
    return pd.DataFrame([(pd.Timestamp(f), t, l, 1.0, 100.0, 0.0, p, m) for f, t, l, p, m in filas], columns=COLUMNAS)


FILAS = [
    ("2022-01-03", "AAA", "compra", np.nan, "entrada_larga"),
    ("2022-01-10", "AAA", "venta", 50.0, "take_profit"),
    ("2022-01-03", "BBB", "venta", np.nan, "entrada_corta"),
    ("2022-01-05", "BBB", "compra", -20.0, "stop_loss"),
    ("2022-01-03", "CCC", "compra", np.nan, "entrada_larga"),
    ("2022-01-04", "CCC", "venta", 5.0, "ajuste"),          # venta parcial por deriva: no es una salida
    ("2022-01-20", "CCC", "venta", -10.0, "stop_loss"),
    ("2022-02-01", "DDD", "venta", np.nan, "entrada_corta"),
    ("2022-02-04", "DDD", "compra", 30.0, "senal_salida"),
]


def test_agrupa_por_motivo_y_lado_con_numero_pnl_acierto_y_duracion():
    filas = {(f["motivo"], f["lado"]): f for f in desglose_por_motivo(_operaciones(FILAS))}
    assert set(filas) == {("take_profit", "largo"), ("stop_loss", "corto"), ("stop_loss", "largo"), ("senal_salida", "corto")}
    assert filas[("take_profit", "largo")] == {"motivo": "take_profit", "lado": "largo", "n": 1, "pnl_total": 50.0, "pnl_medio": 50.0,
                                               "tasa_acierto": 1.0, "duracion_media_dias": 7.0}
    assert filas[("stop_loss", "corto")]["duracion_media_dias"] == 2.0 and filas[("stop_loss", "corto")]["tasa_acierto"] == 0.0
    assert filas[("stop_loss", "largo")]["duracion_media_dias"] == 17.0          # la entrada fue el 3, la salida el 20
    assert filas[("senal_salida", "corto")]["pnl_total"] == 30.0


def test_dos_salidas_del_mismo_motivo_promedian_pnl_acierto_y_duracion():
    filas = FILAS + [("2022-03-01", "AAA", "compra", np.nan, "entrada_larga"), ("2022-03-11", "AAA", "venta", -5.0, "take_profit")]
    fila = next(f for f in desglose_por_motivo(_operaciones(filas)) if f["motivo"] == "take_profit")
    assert fila["n"] == 2 and fila["pnl_total"] == 45.0 and fila["pnl_medio"] == 22.5
    assert fila["tasa_acierto"] == 0.5 and fila["duracion_media_dias"] == pytest.approx(8.5)


def test_lee_operaciones_guardadas_con_el_vocabulario_anterior():
    filas = [("2022-01-03", "AAA", "compra", np.nan, "senal"), ("2022-01-10", "AAA", "venta", 10.0, "stop_perdida"),
             ("2022-01-03", "BBB", "compra", np.nan, "senal"), ("2022-01-11", "BBB", "venta", -4.0, "senal"),
             ("2022-01-03", "CCC", "compra", np.nan, "senal"), ("2022-01-12", "CCC", "venta", 1.0, "fuera_universo")]
    assert {f["motivo"] for f in desglose_por_motivo(_operaciones(filas))} == {"stop_loss", "senal_salida", "salida_universo"}


def test_una_llamada_de_margen_cuenta_como_salida_y_sin_columna_motivo_no_hay_desglose():
    filas = [("2022-01-03", "AAA", "venta", np.nan, "entrada_corta"), ("2022-01-04", "AAA", "compra", -80.0, "llamada_margen")]
    assert [f["motivo"] for f in desglose_por_motivo(_operaciones(filas))] == ["llamada_margen"]
    assert desglose_por_motivo(_operaciones(FILAS).drop(columns="motivo")) == []


def _resultado(estrategia, parametros):
    config = ConfiguracionExperimento(estrategia=estrategia, universo=["AAA", "BBB", "CCC", "DDD", "EEE"], fecha_inicio="2020-01-01",
                                      fecha_fin="2021-12-31", parametros=parametros)
    return ejecutar_con_datos(config, cargar_datos_experimento(config, FuenteSintetica()))


def test_las_metricas_de_una_estrategia_de_senales_traen_el_desglose_y_las_demas_no():
    resultado = _resultado("cruce_medias", {"rapida": 10, "lenta": 40, "lados": "largo_corto", "exposicion_bruta_max": 1.5,
                                            "stop_perdida": 0.05, "max_posiciones": 4})
    desglose = resultado.metricas["motivos_salida"]
    assert desglose and {"largo", "corto"} == {f["lado"] for f in desglose}
    cerradas = resultado.operaciones[resultado.operaciones["pnl"].notna() & (resultado.operaciones["motivo"] != "ajuste")]
    assert sum(f["n"] for f in desglose) == len(cerradas)
    assert sum(f["pnl_total"] for f in desglose) == pytest.approx(cerradas["pnl"].sum())
    assert "motivos_salida" not in _resultado("pesos_iguales", {}).metricas


def test_el_almacen_devuelve_el_desglose_tambien_de_un_experimento_guardado_sin_el(tmp_path):
    from ypyquant.estrategias.almacen import AlmacenExperimentos

    almacen = AlmacenExperimentos(tmp_path / "experimentos")
    config = ConfiguracionExperimento(estrategia="cruce_medias", universo=["AAA", "BBB", "CCC"], fecha_inicio="2020-01-01",
                                      fecha_fin="2021-12-31", parametros={"rapida": 10, "lenta": 40, "stop_perdida": 0.05})
    identificador = almacen.registrar(config)
    almacen.guardar(identificador, ejecutar_con_datos(config, cargar_datos_experimento(config, FuenteSintetica())))
    metricas = almacen.documentos.leer_json(identificador, "metricas")
    del metricas["motivos_salida"]                                   # como un experimento guardado antes de esta métrica
    almacen.documentos.escribir_json(identificador, "metricas", metricas)
    assert almacen.leer(identificador)["metricas_completas"]["motivos_salida"]
