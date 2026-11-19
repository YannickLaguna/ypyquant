"""
Tests de ypyquant.estrategias: registro y descubrimiento, huella, ejecución de experimentos
con precios sintéticos (sin red), almacén Parquet/DuckDB y barridos de rejilla.

Se relaciona con: src/ypyquant/estrategias/*, estrategias_usuario/ y tests/conftest.py.
"""

import pytest
from pydantic import ValidationError

from ypyquant.estrategias import registro
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.base import Estrategia, matriz_cierres, recortar_hasta, sanear_pesos
from ypyquant.estrategias.experimentos import (
    VALIDADORES,
    ConfiguracionExperimento,
    ejecutar_experimento,
    expandir_rejilla,
    registrar_validador,
    resolver_universo,
)
from ypyquant.estrategias.precios import CargadorPreciosCache, CargadorSintetico

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _config(**cambios) -> ConfiguracionExperimento:
    base = {"estrategia": "pesos_iguales", "universo": TICKERS, "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}
    return ConfiguracionExperimento(**{**base, **cambios})


@pytest.fixture
def registro_limpio():
    """Restaura el registro tras tests que registran estrategias temporales."""
    antes = dict(registro._REGISTRO)
    yield
    registro._REGISTRO.clear()
    registro._REGISTRO.update(antes)


# ---- registro y descubrimiento ------------------------------------------------


def test_estrategias_incluidas_registradas_con_schema():
    nombres = {e["nombre"] for e in registro.listar_estrategias()}
    assert {"pesos_iguales", "momentum", "bayes_stein", "michaud", "bl_prior", "preset_ypyquant"} <= nombres
    momentum = registro.descripcion_estrategia("momentum")
    assert momentum["parametros_schema"]["properties"]["top_k"]["default"] == 5
    assert momentum["parametros_defecto"]["meses_retorno"] == 12
    assert "class Momentum" in registro.codigo_fuente("momentum")


def test_estrategia_inexistente_y_nombre_duplicado(registro_limpio):
    with pytest.raises(KeyError, match="no registrada"):
        registro.obtener_estrategia("no_existe")

    class Otra(Estrategia):
        nombre = "pesos_iguales"

        def generar_pesos(self, fecha, datos, estado):
            return {}

    with pytest.raises(ValueError, match="Ya existe"):
        registro.registrar_estrategia(Otra)


def test_parametros_rechazan_campos_desconocidos_y_limites():
    clase = registro.obtener_estrategia("momentum")
    with pytest.raises(ValidationError):
        clase.Parametros(top_k=0)
    with pytest.raises(ValidationError):
        clase.Parametros(inventado=1)


def test_descubrimiento_de_estrategias_de_usuario(tmp_path, registro_limpio):
    (tmp_path / "mi_estrategia.py").write_text(
        "from ypyquant.estrategias import Estrategia, registrar_estrategia\n"
        "@registrar_estrategia\n"
        "class Mia(Estrategia):\n"
        "    nombre = 'mia_prueba'\n"
        "    descripcion = 'de prueba'\n"
        "    def generar_pesos(self, fecha, datos, estado):\n"
        "        return {t: 1.0 for t in datos}\n", encoding="utf-8")
    (tmp_path / "roto.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    (tmp_path / "_ignorado.py").write_text("raise RuntimeError('no se importa')\n", encoding="utf-8")

    nuevos = registro.descubrir_estrategias_usuario(tmp_path)
    assert nuevos == ["mia_prueba"]
    clase = registro.obtener_estrategia("mia_prueba")
    assert clase.origen == "usuario" and clase.archivo.endswith("mi_estrategia.py")
    assert "class Mia" in registro.codigo_fuente("mia_prueba")
    assert any("roto.py" in ruta and "boom" in msg for ruta, msg in registro.ERRORES_DESCUBRIMIENTO.items())
    registro.ERRORES_DESCUBRIMIENTO.clear()


def test_estrategia_de_ejemplo_del_repo_se_descubre():
    assert "inversa_volatilidad" in {e["nombre"] for e in registro.listar_estrategias()}


# ---- adaptador al motor ---------------------------------------------------------


def test_adaptador_recorta_datos_sin_lookahead_y_sanea_pesos():
    datos = {t: CargadorSintetico()(t, "2020-01-01", "2020-12-31") for t in TICKERS[:2]}
    recibidos = []

    class Espia(Estrategia):
        nombre = "espia"

        def generar_pesos(self, fecha, datos, estado):
            recibidos.append(max(df.index.max() for df in datos.values()))
            return {"AAA": 2.0, "BBB": -1.0, "CCC": float("nan")}

    funcion = Espia().como_funcion()
    pesos = funcion(datos["AAA"].index[100].to_pydatetime(), datos)
    assert recibidos[0] <= datos["AAA"].index[100]
    assert pesos == {"AAA": 1.0}  # negativo y NaN descartados, normalizado
    assert sanear_pesos({"A": 0.0}) == {}
    assert len(recortar_hasta(datos, datos["AAA"].index[9])["AAA"]) == 10
    assert list(matriz_cierres(datos).columns) == ["AAA", "BBB"]


# ---- huella ---------------------------------------------------------------------


def test_huella_estable_y_sensible_a_los_cambios():
    a = _config(estrategia="momentum", parametros={"top_k": 3})
    b = _config(estrategia="momentum", parametros={"top_k": 3}, universo=list(reversed(TICKERS)))
    assert a.huella() == b.huella()  # el orden del universo no importa
    assert a.huella() == a.model_copy(update={"etiqueta": "otra"}).huella()  # la etiqueta no entra
    assert a.huella() != _config(estrategia="momentum", parametros={"top_k": 4}).huella()
    assert a.huella() != a.model_copy(update={"semilla": 7}).huella()
    assert len(a.huella()) == 64 and a.identificador().startswith("momentum-")


def test_normalizada_completa_defectos_y_valida():
    n = _config(estrategia="momentum").normalizada()
    assert n.parametros["top_k"] == 5 and n.frecuencia_rebalanceo == "monthly"
    # {} y los defectos explícitos producen la misma huella tras normalizar
    explicita = _config(estrategia="momentum", parametros={"top_k": 5}).normalizada()
    assert n.huella() == explicita.huella()
    with pytest.raises(ValidationError):
        _config(estrategia="momentum", parametros={"top_k": 0}).normalizada()
    with pytest.raises(ValidationError):
        _config(fecha_inicio="2022-01-01", fecha_fin="2021-01-01")


def test_resolver_universo_por_nombre(tmp_path):
    (tmp_path / "mini.txt").write_text("# comentario\nAAA\nbbb\n\nCCC\n", encoding="utf-8")
    assert resolver_universo("mini", tmp_path) == ["AAA", "BBB", "CCC"]
    assert resolver_universo("mini:2", tmp_path) == ["AAA", "BBB"]
    assert resolver_universo(["X"], tmp_path) == ["X"]
    with pytest.raises(ValueError, match="no encontrado"):
        resolver_universo("nada", tmp_path)


# ---- ejecución de experimentos (precios sintéticos) ---------------------------


@pytest.mark.parametrize("estrategia", ["pesos_iguales", "momentum", "bayes_stein", "bl_prior", "inversa_volatilidad"])
def test_ejecutar_experimento_produce_metricas_y_series(estrategia):
    resultado = ejecutar_experimento(_config(estrategia=estrategia), CargadorSintetico())
    m = resultado.metricas
    for clave in ("sharpe", "sortino", "calmar", "max_drawdown", "volatilidad", "turnover_anual", "alfa", "beta", "cagr"):
        assert clave in m and m[clave] is not None
    assert m["max_drawdown"] <= 0 and m["n_rebalanceos"] > 5 and m["validacion"] == {}
    assert {"fecha", "equity", "benchmark", "drawdown", "retorno"} <= set(resultado.equity.columns)
    assert resultado.equity["benchmark"].iloc[0] == pytest.approx(100_000.0)
    assert resultado.pesos.groupby("fecha")["peso"].sum().round(6).eq(1.0).all()
    assert len(resultado.operaciones) == m["n_operaciones"] > 0


def test_ejecutar_experimento_es_reproducible():
    a = ejecutar_experimento(_config(estrategia="michaud", parametros={"n_replicas": 8}, semilla=3), CargadorSintetico())
    b = ejecutar_experimento(_config(estrategia="michaud", parametros={"n_replicas": 8}, semilla=3), CargadorSintetico())
    # Misma semilla → mismas réplicas; el solver (OSQP/BLAS multihilo) puede variar en el
    # último bit, así que se compara con tolerancia relativa 1e-9 y no bit a bit.
    assert a.metricas.keys() == b.metricas.keys()
    for clave, valor in a.metricas.items():
        if isinstance(valor, float):
            assert b.metricas[clave] == pytest.approx(valor, rel=1e-9, abs=1e-12), clave
        else:
            assert b.metricas[clave] == valor, clave


def test_michaud_cambia_con_la_semilla():
    base = {"estrategia": "michaud", "parametros": {"n_replicas": 8}}
    a = ejecutar_experimento(_config(**base, semilla=1), CargadorSintetico())
    b = ejecutar_experimento(_config(**base, semilla=2), CargadorSintetico())
    assert a.metricas["sharpe"] != b.metricas["sharpe"]


def test_preset_ypyquant_envuelve_un_preset():
    config = _config(estrategia="preset_ypyquant", parametros={"preset": "markowitz", "tasa_libre_riesgo": 0.0})
    assert ejecutar_experimento(config, CargadorSintetico()).metricas["n_rebalanceos"] > 5


def test_experimento_sin_precios_falla_con_mensaje_claro():
    with pytest.raises(RuntimeError, match="No hay precios"):
        ejecutar_experimento(_config(), lambda t, i, f: None)


def test_estrategia_que_siempre_falla_aborta_con_su_error(registro_limpio):
    class Rota(Estrategia):
        nombre = "rota_prueba"

        def generar_pesos(self, fecha, datos, estado):
            raise ZeroDivisionError("kaput")

    registro.registrar_estrategia(Rota)
    with pytest.raises(RuntimeError, match="kaput"):
        ejecutar_experimento(_config(estrategia="rota_prueba"), CargadorSintetico())


def test_validador_registrado_se_ejecuta_y_queda_en_metricas():
    registrar_validador("prueba")(lambda resultado: {"n_dias": len(resultado.equity)})
    try:
        r = ejecutar_experimento(_config(validaciones=["prueba"]), CargadorSintetico())
        assert r.metricas["validacion"]["prueba"]["n_dias"] == len(r.equity)
        with pytest.raises(ValueError, match="no registrado"):
            ejecutar_experimento(_config(validaciones=["inexistente"]), CargadorSintetico())
    finally:
        VALIDADORES.pop("prueba", None)


# ---- rejilla ----------------------------------------------------------------------


def test_expandir_rejilla_producto_y_etiquetas():
    base = _config(estrategia="momentum")
    configs = expandir_rejilla(base, {"top_k": [2, 3, 4], "comision": [0.0, 0.002]})
    assert len(configs) == 6 and len({c.huella() for c in configs}) == 6
    assert {c.parametros["top_k"] for c in configs} == {2, 3, 4}
    assert {c.comision for c in configs} == {0.0, 0.002}
    assert "top_k=2" in configs[0].etiqueta
    assert len(expandir_rejilla(base, {"top_k": [3, 3]})) == 1  # duplicados fuera
    assert len(expandir_rejilla(base, {})) == 1
    with pytest.raises(ValueError, match="máximo"):
        expandir_rejilla(base, {"top_k": list(range(1, 60)), "meses_retorno": list(range(2, 12))})
    with pytest.raises(ValidationError):
        expandir_rejilla(base, {"top_k": [0]})


# ---- almacén Parquet + DuckDB ---------------------------------------------------


def test_almacen_ciclo_completo_y_comparacion(tmp_path):
    almacen = AlmacenExperimentos(tmp_path)
    ids = []
    for estrategia in ("pesos_iguales", "momentum"):
        config = _config(estrategia=estrategia).normalizada()
        id_exp = almacen.registrar(config)
        assert almacen.listar(estado="pendiente")[0]["id"] == id_exp
        almacen.marcar_estado(id_exp, "en_curso")
        almacen.guardar(id_exp, ejecutar_experimento(config, CargadorSintetico()))
        ids.append(id_exp)

    for id_exp in ids:
        assert {p.name for p in (tmp_path / id_exp).iterdir()} == {
            "config.json", "estado.json", "metricas.json", "equity.parquet", "pesos.parquet", "operaciones.parquet"}
    listado = almacen.listar()
    assert {r["estado"] for r in listado} == {"hecho"} and listado[0]["metricas"]["sharpe"] is not None
    assert [r["id"] for r in almacen.listar(estrategia="momentum")] == [ids[1]]
    assert almacen.listar(texto="no-aparece") == []

    detalle = almacen.leer(ids[1])
    assert detalle["equity"][0]["fecha"] == "2020-01-01" and detalle["pesos"] and detalle["operaciones"]
    assert detalle["config"]["estrategia"] == "momentum"

    comparacion = almacen.comparar(ids)
    assert comparacion["ids"] == ids and set(comparacion["series"]) == set(ids)
    assert all(serie[0] == pytest.approx(100.0) for serie in comparacion["series"].values())
    assert len(comparacion["fechas"]) == len(comparacion["benchmark"])

    conteo = almacen.ejecutar_sql("select id, count(*) as n from equity group by id order by id")
    assert list(conteo["id"]) == sorted(ids)

    assert almacen.borrar(ids[0]) and not almacen.borrar(ids[0])
    with pytest.raises(KeyError):
        almacen.comparar([ids[0]])


def test_almacen_error_traza_e_interrumpidos(tmp_path):
    almacen = AlmacenExperimentos(tmp_path)
    id_exp = almacen.registrar(_config())
    try:
        raise ValueError("fallo de prueba")
    except ValueError as error:
        almacen.marcar_estado(id_exp, "error", error)
    detalle = almacen.leer(id_exp)
    assert detalle["estado"] == "error" and "fallo de prueba" in detalle["error"] and "Traceback" in detalle["traza"]

    otro = almacen.registrar(_config(estrategia="momentum"))
    almacen.marcar_estado(otro, "en_curso")
    assert almacen.marcar_interrumpidos() == 1 and almacen.leer(otro)["estado"] == "error"
    with pytest.raises(ValueError):
        almacen.leer("../fuera")


# ---- precios ----------------------------------------------------------------------


def test_cargador_cache_usa_parquet_y_no_vuelve_a_descargar(tmp_path, monkeypatch):
    llamadas = []
    sintetico = CargadorSintetico()
    cargador = CargadorPreciosCache(tmp_path)

    def falso_descargar(ticker, inicio, fin):
        llamadas.append((ticker, inicio, fin))
        return sintetico(ticker, inicio, fin)

    monkeypatch.setattr(cargador, "_descargar", falso_descargar)
    primero = cargador("AAA", "2020-01-01", "2020-12-31")
    segundo = cargador("AAA", "2020-03-01", "2020-06-30")
    assert len(llamadas) == 1 and not segundo.empty and segundo.index.min() >= primero.index[0]
    cargador("AAA", "2019-01-01", "2020-12-31")  # rango ampliado: nueva descarga de la unión
    assert len(llamadas) == 2

    sin_red = CargadorPreciosCache(tmp_path, descargar=False)
    assert sin_red("ZZZ", "2020-01-01", "2020-12-31") is None
    assert not sin_red("AAA", "2019-06-01", "2020-12-31").empty
