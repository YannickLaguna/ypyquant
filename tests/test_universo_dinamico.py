"""
test_universo_dinamico.py — Tests del universo point-in-time `sp500_pit` (fase 0.0a-0.0c del plan).

Responsabilidad: comprobar con datos sintéticos y membresías inventadas que
- el filtro del universo deja pasar solo a los miembros del índice en la fecha t con historia hasta t-1
  (un ticker que entra en t+1 NO está en t; uno que sale o deslista después de t SÍ);
- ningún dato con fecha posterior a t influye (`verificar_sin_lookahead`, también con un filtro con
  fuga a propósito, que debe detectarse);
- `Estrategia.como_funcion(filtro_universo=...)` entrega a la estrategia solo los elegibles de cada fecha;
- el motor vende lo que sale del universo y un ticker deslistado se liquida a su último precio, con aviso;
- `sp500_pit` se resuelve como nombre de universo (con ventana de fechas) y se persiste la huella;
- el fichero real `universos/sp500_membresias.json` es coherente (~500 miembros por fecha).
Sin red ni BD.

Se relaciona con: src/ypyquant/estrategias/universo_dinamico.py, base.py, causalidad.py,
experimentos.py y universos/sp500_membresias.json.
"""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from ypyquant.estrategias.base import Estrategia
from ypyquant.estrategias.causalidad import verificar_sin_lookahead
from ypyquant.estrategias.experimentos import (
    ConfiguracionExperimento,
    cargar_datos_experimento,
    ejecutar_con_datos,
    ejecutar_experimento,
    listar_universos,
    resolver_universo,
)
from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.universo_dinamico import (
    NOMBRE_UNIVERSO_PIT,
    UniversoPuntoEnElTiempo,
    cargar_membresias_sp500,
    extraer_membresias_desde_sqlite,
    miembros_en_ventana,
)

FECHAS = pd.bdate_range("2017-01-02", "2021-12-31")
T = pd.Timestamp("2020-06-30")  # fecha de rebalanceo de las pruebas


def _serie(semilla: int, ultimo: str | None = None, primero: str | None = None) -> pd.DataFrame:
    """OHLCV sintético; `ultimo`/`primero` recortan la historia (deslistado / salida a bolsa)."""
    rng = np.random.default_rng(semilla)
    cierre = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(FECHAS))))
    df = pd.DataFrame({"Open": cierre, "High": cierre, "Low": cierre, "Close": cierre,
                       "Volume": 1e6, "adjusted_close": cierre}, index=FECHAS)
    return df.loc[primero:ultimo]


def _intervalo(alta: str | None, baja: str | None) -> list[dict]:
    return [{"index": "SP500", "date_added": alta, "date_removed": baja}]


@pytest.fixture(scope="module")
def datos() -> dict[str, pd.DataFrame]:
    return {
        "AAA": _serie(1),                             # miembro siempre
        "BBB": _serie(2),                             # entra al índice en t+1
        "CCC": _serie(3, ultimo="2020-07-03"),        # sale del índice en t+5 y deslista en t+3
        "DDD": _serie(4, primero="2020-03-02"),       # miembro pero con solo 4 meses de historia en t
        "EEE": _serie(5),                             # nunca fue miembro
    }


@pytest.fixture(scope="module")
def membresias() -> dict[str, list[dict]]:
    return {
        "AAA": _intervalo("1990-01-01", None),
        "BBB": _intervalo("2020-07-01", None),
        "CCC": _intervalo("2015-01-01", "2020-07-05"),
        "DDD": _intervalo("2019-01-01", None),
        "EEE": [],
    }


@pytest.fixture(scope="module")
def universo(membresias) -> UniversoPuntoEnElTiempo:
    return UniversoPuntoEnElTiempo(membresias, meses_historia=12)


# ---- elegibilidad en la fecha t ---------------------------------------------------------------


def test_elegibles_en_t_respeta_membresia_e_historia(universo, datos):
    filtro = universo.crear_filtro(datos)
    assert sorted(filtro(T, datos)) == ["AAA", "CCC"]  # BBB entra después; DDD sin historia; EEE no es miembro


def test_ticker_que_entra_en_t_mas_1_no_esta_en_t_pero_si_en_t_mas_1(universo, datos):
    filtro = universo.crear_filtro(datos)
    assert "BBB" not in filtro(T, datos)
    assert "BBB" in filtro(pd.Timestamp("2020-07-01"), datos)


def test_ticker_que_deslista_en_t_mas_5_si_esta_en_t(universo, datos):
    filtro = universo.crear_filtro(datos)
    assert "CCC" in filtro(T, datos)  # sigue siendo miembro en t aunque deslistará después
    assert "CCC" not in filtro(pd.Timestamp("2020-07-06"), datos)  # ya salió del índice


def test_historia_insuficiente_se_incorpora_cuando_se_cumple(universo, datos):
    filtro = universo.crear_filtro(datos)
    assert "DDD" not in filtro(pd.Timestamp("2020-12-31"), datos)  # 9 meses completos < 12
    assert "DDD" in filtro(pd.Timestamp("2021-06-30"), datos)


def test_el_filtro_devuelve_los_dataframes_recortados_que_recibe(universo, datos):
    filtro = universo.crear_filtro(datos)
    recortados = {t: df.loc[:T] for t, df in datos.items()}
    resultado = filtro(T, recortados)
    assert set(resultado) == {"AAA", "CCC"} and resultado["AAA"] is recortados["AAA"]


def test_con_membresias_vacias_no_hay_elegibles(datos):
    filtro = UniversoPuntoEnElTiempo({}, meses_historia=12).crear_filtro(datos)
    assert filtro(T, datos) == {}


# ---- causalidad ---------------------------------------------------------------------------------


class _PesosIgualesTest(Estrategia):
    nombre = "pesos_iguales_pit_test"

    def generar_pesos(self, fecha, datos, estado):
        return {t: 1 / len(datos) for t in datos} if datos else {}


FECHAS_MUESTRA = [pd.Timestamp(f) for f in ("2020-03-31", "2020-06-30", "2020-07-01", "2020-12-31", "2021-06-30")]


def test_el_universo_pit_no_usa_datos_posteriores_a_t(universo, datos):
    errores = verificar_sin_lookahead(_PesosIgualesTest, {}, datos, FECHAS_MUESTRA, fabrica_filtro_universo=universo.crear_filtro)
    assert errores == []


def test_un_filtro_con_fuga_se_detecta(datos, membresias):
    """Control negativo: exigir que el ticker siga cotizando 30 días después de t (sesgo de supervivencia)."""

    def fabrica_con_fuga(datos_completos):
        base = UniversoPuntoEnElTiempo(membresias, meses_historia=12).crear_filtro(datos_completos)

        def filtro(fecha, recortados):
            sobreviven = {t for t, df in datos_completos.items()
                          if df["Close"].loc[fecha + pd.Timedelta(days=1): fecha + pd.Timedelta(days=30)].notna().any()}
            return {t: df for t, df in base(fecha, recortados).items() if t in sobreviven}

        return filtro

    errores = verificar_sin_lookahead(_PesosIgualesTest, {}, datos, FECHAS_MUESTRA, fabrica_filtro_universo=fabrica_con_fuga)
    assert errores and all(e.startswith("look-ahead en ") for e in errores)  # al envenenar el futuro nadie «sobrevive»


# ---- adaptador de la estrategia -----------------------------------------------------------------


def test_como_funcion_entrega_a_la_estrategia_solo_los_elegibles(universo, datos):
    vistos: list[set[str]] = []

    class Espia(Estrategia):
        nombre = "espia_pit_test"

        def generar_pesos(self, fecha, datos, estado):
            vistos.append(set(datos))
            return {t: 1 / len(datos) for t in datos}

    funcion = Espia().como_funcion(filtro_universo=universo.crear_filtro(datos))
    funcion(T, datos)
    assert vistos == [{"AAA", "CCC"}]
    # Sin filtro el comportamiento es el de siempre: todos los tickers con datos
    vistos.clear()
    Espia().como_funcion()(T, datos)
    assert vistos == [set(datos)]


# ---- membresías: fuente y utilidades ---------------------------------------------------------------


def test_miembros_en_ventana_incluye_los_que_se_solapan(membresias):
    assert miembros_en_ventana(membresias, "2020-07-02", "2020-12-31") == ["AAA", "BBB", "CCC", "DDD"]
    assert miembros_en_ventana(membresias, "2020-07-06", "2020-12-31") == ["AAA", "BBB", "DDD"]
    assert miembros_en_ventana(membresias, "2000-01-01", "2000-12-31") == ["AAA"]


def test_extraer_membresias_desde_sqlite_filtra_indice_y_normaliza_tickers(tmp_path):
    ruta = tmp_path / "mini.db"
    conn = sqlite3.connect(ruta)
    conn.execute("CREATE TABLE tickers (symbol TEXT, index_memberships TEXT)")
    conn.executemany("INSERT INTO tickers VALUES (?, ?)", [
        ("BRK.B", json.dumps([{"index": "SP500", "date_added": "2010-02-16T00:00:00", "date_removed": None,
                               "reason_removed": None}])),
        ("QQQ", json.dumps([{"index": "NASDAQ", "date_added": "2026-03-16", "date_removed": None}])),
        ("XYZ", None),
        ("OLD", json.dumps([{"index": "SP500", "date_added": "2001-01-01", "date_removed": "2005-05-05T00:00:00"},
                            {"index": "DOW_JONES", "date_added": "2003-01-01", "date_removed": None}])),
    ])
    conn.commit()
    conn.close()
    assert extraer_membresias_desde_sqlite(ruta) == {
        "BRK-B": [{"date_added": "2010-02-16", "date_removed": None}],
        "OLD": [{"date_added": "2001-01-01", "date_removed": "2005-05-05"}],
    }


def test_fichero_real_de_membresias_es_coherente():
    """~500 miembros del S&P 500 en cada fecha desde 2000 (la fuente no es fiable antes de 1999)."""
    membresias = cargar_membresias_sp500()
    assert len(membresias) > 1000
    for fecha in ("2005-06-30", "2012-12-31", "2018-06-29", "2023-12-29"):
        n = len(miembros_en_ventana(membresias, fecha, fecha))
        assert 470 <= n <= 520, (fecha, n)
    assert "BRK-B" in membresias and "BRK.B" not in membresias  # formato de yfinance


# ---- integración con los experimentos ---------------------------------------------------------------


def test_sp500_pit_es_un_universo_listable_y_resoluble():
    nombres = {u["nombre"]: u for u in listar_universos()}
    assert NOMBRE_UNIVERSO_PIT in nombres and nombres[NOMBRE_UNIVERSO_PIT]["dinamico"] is True
    todos = resolver_universo(NOMBRE_UNIVERSO_PIT)
    ventana = resolver_universo(NOMBRE_UNIVERSO_PIT, inicio="2020-01-01", fin="2020-12-31")
    assert set(ventana) < set(todos) and "AAPL" in ventana
    with pytest.raises(ValueError, match="no admite"):
        resolver_universo(f"{NOMBRE_UNIVERSO_PIT}:30")


def test_la_huella_de_experimentos_con_universo_fijo_no_cambia_y_la_pit_es_distinta():
    base = {"estrategia": "momentum", "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}
    fijo = ConfiguracionExperimento(universo="sp500:30", **base)
    pit = ConfiguracionExperimento(universo=NOMBRE_UNIVERSO_PIT, **base)
    assert fijo.huella() != pit.huella()
    assert pit.model_dump()["universo"] == NOMBRE_UNIVERSO_PIT


# ---- motor: lo que sale del universo se vende; los deslistados se liquidan a su último precio ------------


@pytest.fixture
def con_membresias_de_prueba(monkeypatch, membresias):
    """Sustituye la fuente de membresías por la de prueba (más un miembro sin precios, FFF)."""
    from ypyquant.estrategias import experimentos

    completas = {t: [{"index": "SP500", **m} for m in intervalos]
                 for t, intervalos in {**membresias, "FFF": _intervalo("2015-01-01", None)}.items()}
    monkeypatch.setattr(experimentos, "cargar_membresias_sp500", lambda: completas)
    return completas


def _ejecutar_pit(datos):
    from ypyquant.estrategias.experimentos import DatosExperimento

    config = ConfiguracionExperimento(estrategia="pesos_iguales", universo=NOMBRE_UNIVERSO_PIT,
                                      fecha_inicio="2020-01-02", fecha_fin="2020-12-31", comision=0.0, slippage=0.0)
    return ejecutar_con_datos(config, DatosExperimento(datos, None, config.benchmark))


def test_el_motor_solo_recibe_pesos_de_los_elegibles_de_cada_fecha(con_membresias_de_prueba, datos):
    resultado = _ejecutar_pit(datos)
    por_fecha = resultado.pesos.groupby("fecha")["ticker"].apply(set)
    antes = por_fecha[por_fecha.index < "2020-07-01"]
    despues = por_fecha[por_fecha.index >= "2020-08-01"]
    assert all(s == {"AAA", "CCC"} for s in antes)  # DDD sin historia, BBB aún no entra, EEE nunca
    assert all(s == {"AAA", "BBB"} for s in despues)  # CCC salió del índice; BBB entró; DDD aún no cumple 12 meses
    assert por_fecha[pd.Timestamp("2020-07-01")] == {"AAA", "BBB", "CCC"}  # BBB entra el día t; CCC sigue (sale en t+4)


def test_la_posicion_que_sale_del_universo_se_vende_y_el_deslistado_avisa(con_membresias_de_prueba, datos):
    resultado = _ejecutar_pit(datos)
    ventas_ccc = resultado.operaciones[(resultado.operaciones["ticker"] == "CCC") & (resultado.operaciones["lado"] == "venta")]
    liquidacion = ventas_ccc.iloc[-1]  # las ventas previas son ajustes de peso; la última cierra la posición
    assert liquidacion["fecha"] == pd.Timestamp("2020-08-03")
    assert liquidacion["precio"] == pytest.approx(datos["CCC"]["Close"].iloc[-1])  # a su último precio conocido (sin slippage aquí)
    assert any(a.startswith("CCC se liquidó a su último precio conocido (2020-07-03)") for a in resultado.avisos)


def test_el_aviso_de_cobertura_cuenta_los_miembros_sin_precio(con_membresias_de_prueba, datos):
    resultado = _ejecutar_pit(datos)
    aviso = next(a for a in resultado.avisos if a.startswith("Universo sp500_pit"))
    assert "1 de 5 miembros históricos" in aviso  # FFF es miembro y no tiene precios (EEE no es miembro)
    assert "sesgo de supervivencia" in aviso
    assert sum(a.startswith("Universo sp500_pit") for a in resultado.avisos) == 1


def test_ejecutar_experimento_completo_con_universo_pit(con_membresias_de_prueba):
    """Camino completo (carga con el cargador sintético, métricas, procedencia) con universo `sp500_pit`."""
    config = ConfiguracionExperimento(estrategia="momentum", parametros={"top_k": 1, "meses_retorno": 6},
                                      universo=NOMBRE_UNIVERSO_PIT, fecha_inicio="2020-01-02", fecha_fin="2020-12-31")
    datos_carga = cargar_datos_experimento(config, FuenteSintetica())
    assert set(datos_carga.precios) == {"AAA", "BBB", "CCC", "DDD", "FFF"}  # el sintético sirve precios a todos
    resultado = ejecutar_experimento(config, FuenteSintetica())
    assert resultado.metricas["n_rebalanceos"] >= 10
    assert sum(a.startswith("Universo sp500_pit") for a in resultado.avisos) == 1
