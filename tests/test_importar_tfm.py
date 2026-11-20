"""
test_importar_tfm.py — Tests de la importación de los resultados del TFM al almacén de experimentos.

Responsabilidad: comprobar con Parquet sintéticos pequeños (sin depender de los datos reales) los ids,
la serie promediada, las métricas, los avisos y la procedencia de los experimentos importados, la
idempotencia (`forzar`), y que `listar`/`leer` y `GET /api/comparar` (TestClient, como en
tests/test_api.py) funcionan con un experimento importado y uno normal a la vez. Un test marcado
`datos` usa los Parquet reales del TFM si existen (y se salta por defecto: `pytest -m datos`).

Se relaciona con: src/ypyquant/estrategias/importacion_tfm.py, scripts/importar_tfm.py,
ypyquant.estrategias.almacen, app/api (comparación) y tests/test_paridad_tfm.py (mismo patrón `datos`).
"""

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import crear_app
from app.api.paper import BotControl, ServicioAlpaca
from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.importacion_tfm import (
    AVISO_SESGOS,
    ErrorImportacionTfm,
    identificador_tfm,
    importar_experimentos_tfm,
)
from ypyquant.estrategias.precios import CargadorSintetico

MESES = pd.date_range("2010-01-01", periods=24, freq="MS")
ESTRATEGIAS = ["mpt_muestral", "igual_ponderacion"]
CELDAS = [(20, 12), (40, 24)]
TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def _cargador_spy(ticker: str, inicio: str, fin: str) -> pd.DataFrame:
    """Precios diarios sintéticos de 'SPY' desde antes del primer mes (el importador necesita el mes previo)."""
    fechas = pd.bdate_range("2009-11-02", "2012-01-31")
    semilla = np.random.default_rng(7)
    cierre = 100 * np.cumprod(1 + semilla.normal(0.0004, 0.01, len(fechas)))
    return pd.DataFrame({"Close": cierre, "adjusted_close": cierre}, index=fechas).loc[inicio:fin]


@pytest.fixture
def carpeta_tfm(tmp_path: Path) -> Path:
    """Mini `rejilla_v2` sintética: 3 réplicas x 2 N x 2 T x 2 estrategias x 24 meses (+ pesos y rf)."""
    carpeta = tmp_path / "rejilla_v2"
    carpeta.mkdir()
    generador = np.random.default_rng(1)
    filas = []
    for b in range(3):
        for n, t in CELDAS:
            for estrategia in ESTRATEGIAS:
                r_bruto = generador.normal(0.01, 0.04, len(MESES))
                turnover = np.concatenate([[0.0], generador.uniform(0.0, 0.3, len(MESES) - 1)])
                tc = 0.001 * turnover
                filas.append(pd.DataFrame({
                    "b": b, "N": n, "T": t, "estrategia": estrategia, "fecha": MESES, "r_bruto": r_bruto,
                    "turnover": turnover, "tc": tc, "r_neto": (1 + r_bruto) * (1 - tc) - 1}))
    pd.concat(filas, ignore_index=True).to_parquet(carpeta / "resultados_oos.parquet")
    indice = pd.MultiIndex.from_tuples([(n, t, e) for n, t in CELDAS for e in ESTRATEGIAS], names=["N", "T", "estrategia"])
    pd.DataFrame({"sharpe_media_replicas": 0.5, "sharpe_serie_promediada": 0.9, "volatilidad": 0.15,
                  "retorno_anual": 0.1, "max_drawdown": -0.2}, index=indice).to_parquet(carpeta / "metricas_por_celda.parquet")
    rf = pd.DataFrame({"rf": 0.0003}, index=pd.date_range("2009-01-01", periods=40, freq="MS", name="fecha"))
    rf.to_parquet(carpeta / "rf_mensual.parquet")
    pesos = pd.DataFrame({
        "b": 0, "N": 20, "T": 12, "estrategia": "mpt_muestral", "fecha": np.repeat(MESES, 2),
        "ticker": ["AAA", "BBB"] * len(MESES), "w": 0.5})
    pesos.to_parquet(carpeta / "pesos_oos.parquet")
    return carpeta


def _importar(carpeta_tfm: Path, almacen: AlmacenExperimentos, **opciones):
    opciones.setdefault("celdas", CELDAS)
    opciones.setdefault("estrategias", ESTRATEGIAS)
    opciones.setdefault("cargador_precios", _cargador_spy)
    return importar_experimentos_tfm(carpeta_tfm, almacen, **opciones)


def test_ids_estables_y_legibles():
    assert identificador_tfm("mpt_bayes_stein", 80, 36) == "tfm-mpt_bayes_stein-n80-t36-media"
    assert identificador_tfm("bl_prior", 20, 12, replica=3) == "tfm-bl_prior-n20-t12-b3"


def test_importa_todas_las_celdas_y_estrategias(carpeta_tfm, tmp_path):
    almacen = AlmacenExperimentos(tmp_path / "exp")
    acciones = _importar(carpeta_tfm, almacen)
    ids = {a["id"] for a in acciones}
    assert len(ids) == 4 and {a["accion"] for a in acciones} == {"importado"}
    assert "tfm-mpt_muestral-n20-t12-media" in ids and "tfm-igual_ponderacion-n40-t24-media" in ids
    for i in ids:
        assert almacen.estado_de(i) == "hecho"  # el id pasa la validación del almacén


def test_serie_promediada_y_equity(carpeta_tfm, tmp_path):
    almacen = AlmacenExperimentos(tmp_path / "exp")
    _importar(carpeta_tfm, almacen)
    original = pd.read_parquet(carpeta_tfm / "resultados_oos.parquet")
    celda = original[(original.N == 20) & (original["T"] == 12) & (original.estrategia == "mpt_muestral")]
    media = celda.groupby("fecha")["r_neto"].mean().to_numpy()

    equity = almacen.leer_equity("tfm-mpt_muestral-n20-t12-media")
    assert len(equity) == len(MESES) + 1 and equity["equity"].iloc[0] == 100_000.0  # punto base + 24 meses
    np.testing.assert_allclose(equity["retorno"].to_numpy()[1:], media)
    np.testing.assert_allclose(equity["equity"].iloc[-1], 100_000 * np.prod(1 + media))
    np.testing.assert_allclose(equity["drawdown"].min(), (equity["equity"] / equity["equity"].cummax() - 1).min())
    # las fechas son cierres de mes (último día de negociación de SPY), no el primer día del TFM
    assert equity["fecha"].iloc[1] == pd.Timestamp("2010-01-29") and equity["fecha"].iloc[0] == pd.Timestamp("2009-12-31")
    assert equity["benchmark"].notna().all() and equity["benchmark"].iloc[0] == 100_000.0


def test_metricas_con_nombres_normales_y_las_del_tfm(carpeta_tfm, tmp_path):
    almacen = AlmacenExperimentos(tmp_path / "exp")
    _importar(carpeta_tfm, almacen)
    metricas = almacen.leer_metricas("tfm-mpt_muestral-n20-t12-media")
    for clave in ("retorno_total", "cagr", "volatilidad", "sharpe", "sortino", "calmar", "max_drawdown",
                  "turnover_anual", "alfa", "beta", "ulcer", "martin", "capital_final"):
        assert metricas[clave] is not None, clave
    media = pd.read_parquet(carpeta_tfm / "resultados_oos.parquet").query(
        "N == 20 and T == 12 and estrategia == 'mpt_muestral'").groupby("fecha")["r_neto"].mean()
    assert metricas["sharpe_media_desv"] == pytest.approx(np.sqrt(12) * media.mean() / media.std(ddof=1))
    assert metricas["sharpe_exceso_rf"] == pytest.approx(np.sqrt(12) * (media - 0.0003).mean() / (media - 0.0003).std(ddof=1))
    assert metricas["max_drawdown"] <= 0 and metricas["capital_final"] == pytest.approx(100_000 * (1 + media).prod())
    # métricas originales tal cual, para trazabilidad
    assert metricas["tfm_sharpe_serie_promediada"] == pytest.approx(0.9)
    assert metricas["tfm_sharpe_media_replicas"] == pytest.approx(0.5)


def test_config_procedencia_y_avisos(carpeta_tfm, tmp_path):
    almacen = AlmacenExperimentos(tmp_path / "exp")
    _importar(carpeta_tfm, almacen)
    detalle = almacen.leer("tfm-igual_ponderacion-n40-t24-media")
    config = detalle["config"]
    assert config["estrategia"] == "tfm_igual_ponderacion" and config["parametros"] == {"N": 40, "T": 24, "replica": "media_3"}
    assert config["universo"] == "tfm:N=40" and config["frecuencia_rebalanceo"] == "monthly"
    assert (config["fecha_inicio"], config["fecha_fin"]) == ("2010-01-01", "2011-12-31")
    assert config["comision"] == 0.001 and config["slippage"] == 0.0
    assert detalle["etiqueta"] == "TFM original (rejilla_v2) N=40 T=24 · media 3 réplicas"
    proc = detalle["procedencia"]
    assert proc["origen"] == "importado del TFM" and proc["importado"]
    assert set(proc["archivos"]) == {"resultados_oos.parquet", "metricas_por_celda.parquet", "rf_mensual.parquet"}
    assert all(len(a["sha256"]) == 64 for a in proc["archivos"].values())
    avisos = detalle["metricas_completas"]["avisos"]
    assert AVISO_SESGOS in avisos and any("media de r_neto" in a for a in avisos)
    assert detalle["pesos"] == [] and detalle["operaciones"] == []  # serie promediada: sin carteras


def test_sin_benchmark_se_avisa(carpeta_tfm, tmp_path):
    almacen = AlmacenExperimentos(tmp_path / "exp")
    _importar(carpeta_tfm, almacen, cargador_precios=None)
    detalle = almacen.leer("tfm-mpt_muestral-n20-t12-media")
    assert all(p["benchmark"] is None for p in detalle["equity"])
    assert any("Sin benchmark SPY" in a for a in detalle["metricas_completas"]["avisos"])
    assert "alfa" not in detalle["metricas_completas"]


def test_benchmark_que_no_cubre_el_periodo_se_ignora(carpeta_tfm, tmp_path):
    almacen = AlmacenExperimentos(tmp_path / "exp")
    _importar(carpeta_tfm, almacen, cargador_precios=CargadorSintetico())  # empieza en 2010-01-04: falta 2009-12
    detalle = almacen.leer("tfm-mpt_muestral-n20-t12-media")
    assert any("no cubren todo el periodo" in a for a in detalle["metricas_completas"]["avisos"])


def test_reimportar_es_idempotente_y_forzar_sobrescribe(carpeta_tfm, tmp_path):
    almacen = AlmacenExperimentos(tmp_path / "exp")
    _importar(carpeta_tfm, almacen)
    segunda = _importar(carpeta_tfm, almacen)
    assert {a["accion"] for a in segunda} == {"omitido"}
    marca = tmp_path / "exp" / "tfm-mpt_muestral-n20-t12-media" / "marca.txt"
    marca.write_text("resto")
    tercera = _importar(carpeta_tfm, almacen, forzar=True)
    assert {a["accion"] for a in tercera} == {"sobrescrito"} and not marca.exists()


def test_replica_concreta_con_pesos(carpeta_tfm, tmp_path):
    almacen = AlmacenExperimentos(tmp_path / "exp")
    acciones = _importar(carpeta_tfm, almacen, celdas=[(20, 12)], estrategias=["mpt_muestral"], replica=0)
    assert acciones == [{"id": "tfm-mpt_muestral-n20-t12-b0", "accion": "importado"}]
    detalle = almacen.leer("tfm-mpt_muestral-n20-t12-b0")
    assert detalle["parametros"]["replica"] == 0 and "réplica 0" in detalle["etiqueta"]
    assert len(detalle["pesos"]) == 2 * len(MESES) and {p["ticker"] for p in detalle["pesos"]} == {"AAA", "BBB"}
    assert detalle["pesos"][0]["fecha"] == "2010-01-29"
    original = pd.read_parquet(carpeta_tfm / "resultados_oos.parquet").query(
        "b == 0 and N == 20 and T == 12 and estrategia == 'mpt_muestral'")
    np.testing.assert_allclose([p["retorno"] for p in detalle["equity"][1:]], original["r_neto"].to_numpy())
    assert "pesos_oos.parquet" in detalle["procedencia"]["archivos"]


def test_errores_claros(carpeta_tfm, tmp_path):
    almacen = AlmacenExperimentos(tmp_path / "exp")
    with pytest.raises(ErrorImportacionTfm, match="no existen"):
        _importar(carpeta_tfm, almacen, celdas=[(99, 99)])
    with pytest.raises(ErrorImportacionTfm, match="desconocidas"):
        _importar(carpeta_tfm, almacen, estrategias=["inventada"])
    with pytest.raises(ErrorImportacionTfm, match="Faltan archivos"):
        _importar(tmp_path, almacen)


def _esperar(cliente: TestClient, id_experimento: str, timeout: float = 60.0) -> dict:
    limite = time.time() + timeout
    while time.time() < limite:
        detalle = cliente.get(f"/api/experimentos/{id_experimento}").json()
        if detalle["estado"] in ("hecho", "error"):
            return detalle
        time.sleep(0.1)
    raise AssertionError("El experimento no terminó a tiempo")


def test_listar_leer_y_comparar_con_un_experimento_normal(carpeta_tfm, tmp_path, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    raiz = tmp_path / "exp"
    app = crear_app(CargadorSintetico(), raiz, ServicioAlpaca(), BotControl(tmp_path / "paper"), cargar_env=False)
    cliente = TestClient(app)
    normal = cliente.post("/api/experimentos", json={
        "estrategia": "pesos_iguales", "universo": TICKERS, "fecha_inicio": "2010-03-01", "fecha_fin": "2011-12-31",
        "frecuencia_rebalanceo": "monthly"}).json()["id"]
    assert _esperar(cliente, normal)["estado"] == "hecho"
    _importar(carpeta_tfm, AlmacenExperimentos(raiz), celdas=[(20, 12)], estrategias=["mpt_muestral"])
    importado = "tfm-mpt_muestral-n20-t12-media"

    lista = cliente.get("/api/experimentos").json()["experimentos"]
    assert {e["id"] for e in lista} == {normal, importado}
    resumen = next(e for e in lista if e["id"] == importado)
    assert resumen["estado"] == "hecho" and resumen["estrategia"] == "tfm_mpt_muestral" and resumen["metricas"]["sharpe"] is not None
    assert cliente.get("/api/experimentos", params={"texto": "TFM original"}).json()["experimentos"][0]["id"] == importado
    detalle = cliente.get(f"/api/experimentos/{importado}").json()
    assert len(detalle["equity"]) == len(MESES) + 1 and detalle["metricas_completas"]["avisos"]

    comparacion = cliente.get("/api/comparar", params={"ids": f"{normal},{importado}"}).json()
    assert comparacion["ids"] == [normal, importado]
    assert len(comparacion["fechas"]) >= 12  # cierres de mes comunes entre la serie mensual y la diaria
    assert all(abs(serie[0] - 100.0) < 1e-6 for serie in comparacion["series"].values())
    assert comparacion["resumenes"][importado]["estrategia"] == "tfm_mpt_muestral"
    assert comparacion["metricas"][importado]["sharpe"] is not None
    # solo el importado también se compara y se borra como cualquier otro
    assert cliente.get("/api/comparar", params={"ids": importado}).status_code == 200
    assert cliente.post("/api/experimentos/borrar", json={"ids": [importado]}).json()["borrados"] == [importado]


def test_cli_importa_en_un_almacen_temporal(carpeta_tfm, tmp_path, capsys):
    ruta = Path(__file__).resolve().parent.parent / "scripts" / "importar_tfm.py"
    especificacion = importlib.util.spec_from_file_location("importar_tfm", ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    codigo = modulo.main(["--origen", str(carpeta_tfm), "--almacen", str(tmp_path / "exp"), "--celda", "20,12",
                          "--estrategia", "mpt_muestral", "--sin-benchmark"])
    assert codigo == 0 and "tfm-mpt_muestral-n20-t12-media" in capsys.readouterr().out
    assert (tmp_path / "exp" / "tfm-mpt_muestral-n20-t12-media" / "metricas.json").exists()
    assert modulo.main(["--origen", str(carpeta_tfm), "--almacen", str(tmp_path / "exp"), "--celda", "1,2"]) == 1


@pytest.mark.datos
def test_datos_reales_sharpe_serie_promediada(tmp_path):
    """Con los Parquet reales: el Sharpe √12·media/desv. de la serie importada reproduce `sharpe_serie_promediada`."""
    origen = Path(os.environ.get("TFM_DIR", "C:/Users/yanni/Documents/Codigo/Proyectos/tfm")) / "data" / "reportes" / "rejilla_v2"
    if not (origen / "resultados_oos.parquet").exists():
        pytest.skip("Sin los datos del TFM")
    almacen = AlmacenExperimentos(tmp_path / "exp")
    acciones = importar_experimentos_tfm(origen, almacen, cargador_precios=None)
    assert len(acciones) == 18
    for accion in acciones:
        metricas = almacen.leer_metricas(accion["id"])
        assert metricas["sharpe_media_desv"] == pytest.approx(metricas["tfm_sharpe_serie_promediada"], abs=1e-6), accion["id"]
        json.dumps(metricas)  # JSON válido
