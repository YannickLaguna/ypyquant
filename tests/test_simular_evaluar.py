"""
test_simular_evaluar.py — Interfaz `simular` -> `evaluar`, reevaluación desde Parquet y semillas por ejecución.

Responsabilidad:
    - `simular` devuelve solo hechos (`Simulacion`: equity, pesos, operaciones, rotaciones, avisos); `evaluar`
      calcula métricas, y su composición equivale a `ejecutar_con_datos`.
    - `evaluar` acepta cualquier simulación construida desde tablas (`simulacion_desde_tablas`): es la costura
      por la que entrará el paper interno.
    - `AlmacenExperimentos.reevaluar` y `scripts/reevaluar_experimentos.py` recalculan `metricas.json` solo con los
      Parquet, sin repetir el backtest.
    - Cada ejecución usa su propio generador: no se toca el estado global de `random`/`np.random` y dos
      experimentos con aleatoriedad en hilos distintos dan lo mismo que en serie.

Se relaciona con:
    src/ypyquant/estrategias/{simulacion,evaluacion,experimentos,almacen,base,incluidas}.py y
    scripts/reevaluar_experimentos.py; tests/test_caracterizacion_metricas.py fija los valores numéricos.
"""

import importlib.util
import random
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ypyquant.estrategias.almacen import AlmacenExperimentos
from ypyquant.estrategias.base import Estrategia
from ypyquant.estrategias.evaluacion import Evaluacion, evaluar
from ypyquant.estrategias.experimentos import (
    ConfiguracionExperimento,
    cargar_datos_experimento,
    ejecutar_con_datos,
    ejecutar_experimento,
)
from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.registro import registrar_estrategia
from ypyquant.estrategias.simulacion import Simulacion, simular, simulacion_desde_tablas

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _config(**cambios) -> ConfiguracionExperimento:
    base = {"estrategia": "momentum", "parametros": {"top_k": 3, "meses_retorno": 6}, "universo": TICKERS,
            "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}
    return ConfiguracionExperimento(**{**base, **cambios})


def _config_michaud(semilla: int) -> ConfiguracionExperimento:
    return _config(estrategia="michaud", parametros={"n_replicas": 8}, semilla=semilla)


@pytest.fixture(scope="module")
def datos():
    return cargar_datos_experimento(_config(), FuenteSintetica())


# ---- interfaz ----------------------------------------------------------------


def test_simular_devuelve_hechos_sin_metricas(datos):
    simulacion = simular(_config(), datos)
    assert isinstance(simulacion, Simulacion)
    assert list(simulacion.equity.columns) == ["fecha", "equity", "drawdown", "retorno"]
    assert list(simulacion.rotaciones.columns) == ["fecha", "rotacion"]
    assert len(simulacion.pesos) and len(simulacion.operaciones)
    assert simulacion.procedencia["hash_precios"]
    assert not hasattr(simulacion, "metricas")


def test_simular_sin_completo_omite_procedencia(datos):
    assert simular(_config(), datos, completo=False).procedencia == {}


def test_evaluar_de_una_simulacion_equivale_a_ejecutar_con_datos(datos):
    config = _config(validaciones=["rachas", "psr"])
    evaluacion = evaluar(simular(config, datos))
    resultado = ejecutar_con_datos(config, datos)
    assert isinstance(evaluacion, Evaluacion)
    assert evaluacion.metricas == resultado.metricas
    pd.testing.assert_frame_equal(evaluacion.equity, resultado.equity)
    assert set(evaluacion.metricas["validacion"]) == {"rachas", "psr"}


def test_evaluar_sin_validar_omite_validadores(datos):
    evaluacion = evaluar(simular(_config(validaciones=["psr"]), datos), validar=False)
    assert evaluacion.metricas["validacion"] == {}


def test_evaluar_acepta_cualquier_simulacion_construida_desde_tablas(datos):
    """El paper interno construirá la simulación desde lo ocurrido: mismas métricas que la original."""
    original = simular(_config(), datos)
    ajena = simulacion_desde_tablas(original.config, original.equity, original.pesos, original.operaciones,
                                    original.rotaciones, benchmark_precios=original.benchmark_precios)
    assert evaluar(ajena).metricas == evaluar(original).metricas


# ---- reevaluación desde Parquet ------------------------------------------------


def _guardar(tmp_path, datos, config=None) -> tuple[AlmacenExperimentos, str, dict]:
    almacen = AlmacenExperimentos(tmp_path)
    config = config or _config(validaciones=["psr"])
    id_experimento = almacen.registrar(config)
    almacen.guardar(id_experimento, ejecutar_con_datos(config, datos))
    return almacen, id_experimento, almacen.leer_metricas(id_experimento)


def _casi_iguales(nuevas: dict, antes: dict, excepto: tuple[str, ...] = ()) -> None:
    for clave, valor in antes.items():
        if clave in excepto or clave in ("avisos", "reevaluacion"):
            continue
        if isinstance(valor, dict):
            _casi_iguales(nuevas[clave], valor)
        elif isinstance(valor, float):
            assert nuevas[clave] == pytest.approx(valor, rel=1e-9, abs=1e-12), clave
        else:
            assert nuevas[clave] == valor, clave


def test_reevaluar_reproduce_las_metricas_guardadas_sin_repetir_el_backtest(tmp_path, datos, monkeypatch):
    almacen, id_experimento, antes = _guardar(tmp_path, datos)
    assert (tmp_path / id_experimento / "rotaciones.parquet").exists()
    monkeypatch.setattr("ypyquant.estrategias.simulacion.BacktestEngine",
                        lambda *a, **k: pytest.fail("no debe repetir el backtest"))
    nuevas = almacen.reevaluar(id_experimento)
    _casi_iguales(nuevas, antes)
    assert almacen.leer_metricas(id_experimento)["avisos"] == antes["avisos"]


def test_reevaluar_sin_rotaciones_estima_el_turnover_y_lo_marca(tmp_path, datos):
    almacen, id_experimento, antes = _guardar(tmp_path, datos)
    (tmp_path / id_experimento / "rotaciones.parquet").unlink()
    nuevas = almacen.reevaluar(id_experimento, guardar=False)
    _casi_iguales(nuevas, antes, excepto=("turnover_medio", "turnover_anual"))
    assert "reevaluacion" in nuevas
    assert nuevas["turnover_anual"] == pytest.approx(antes["turnover_anual"], rel=0.25)
    assert "reevaluacion" not in almacen.leer_metricas(id_experimento)  # guardar=False no escribe


def test_reevaluar_omite_importaciones_y_no_terminados(tmp_path, datos):
    almacen, id_experimento, _ = _guardar(tmp_path, datos)
    pendiente = almacen.registrar(_config(estrategia="pesos_iguales"))
    with pytest.raises(ValueError, match="no está terminado"):
        almacen.reevaluar(pendiente)
    (tmp_path / id_experimento / "procedencia.json").write_text('{"origen": "importado del TFM"}', encoding="utf-8")
    with pytest.raises(ValueError, match="importación"):
        almacen.reevaluar(id_experimento)


def test_script_reevaluar_experimentos_sobre_una_copia(tmp_path, datos, capsys):
    _, id_experimento, antes = _guardar(tmp_path / "original", datos)
    copia = tmp_path / "copia"
    shutil.copytree(tmp_path / "original", copia)
    ruta = Path(__file__).resolve().parents[1] / "scripts" / "reevaluar_experimentos.py"
    spec = importlib.util.spec_from_file_location("reevaluar_experimentos", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    metricas_json = copia / id_experimento / "metricas.json"
    metricas_json.write_text(metricas_json.read_text(encoding="utf-8").replace('"sharpe"', '"sharpe_viejo"'), encoding="utf-8")
    assert modulo.main(["--raiz", str(copia), "--simular"]) == 0
    assert "sharpe" in capsys.readouterr().out
    assert modulo.main(["--raiz", str(copia)]) == 0
    assert AlmacenExperimentos(copia).leer_metricas(id_experimento)["sharpe"] == pytest.approx(antes["sharpe"], rel=1e-9)


# ---- semillas por ejecución ------------------------------------------------------


def test_simular_no_toca_el_estado_global_de_azar(datos):
    random.seed(7)
    np.random.seed(7)
    estado_np, estado_py = np.random.get_state(), random.getstate()
    simular(_config_michaud(3), datos)
    assert random.getstate() == estado_py
    nuevo = np.random.get_state()
    assert nuevo[0] == estado_np[0] and (nuevo[1] == estado_np[1]).all() and nuevo[2:] == estado_np[2:]


def test_rng_explicito_equivale_a_la_semilla_de_la_configuracion(datos):
    config = _config_michaud(3)
    por_defecto = evaluar(simular(config, datos)).metricas
    explicito = evaluar(simular(config, datos, rng=np.random.default_rng(3))).metricas
    otro = evaluar(simular(config, datos, rng=np.random.default_rng(4))).metricas
    assert por_defecto == explicito and por_defecto != otro


def test_michaud_en_hilos_distintos_no_interfiere(datos):
    """La API usa un ThreadPoolExecutor de 2: lo concurrente debe dar lo mismo que lo secuencial."""
    configs = [_config_michaud(1), _config_michaud(2)]
    en_serie = [ejecutar_con_datos(c, datos).metricas for c in configs]
    barrera = threading.Barrier(2)

    def correr(config):
        barrera.wait()  # arrancan a la vez para forzar el solapamiento
        return ejecutar_con_datos(config, datos).metricas

    for _ in range(2):
        with ThreadPoolExecutor(max_workers=2) as pool:
            en_paralelo = list(pool.map(correr, configs))
        assert en_paralelo == en_serie
    assert en_serie[0] != en_serie[1]  # las semillas distintas dan resultados distintos (el test es sensible)


def test_estrategia_de_usuario_con_azar_usa_generador_propio_y_es_reproducible(datos):
    """Una estrategia que sortea con `generador_azar()` es determinista por semilla y concurrente sin interferir."""

    @registrar_estrategia
    class SorteoPrueba(Estrategia):
        nombre = "sorteo_prueba_simular"
        frecuencia_rebalanceo = "monthly"

        def generar_pesos(self, fecha, datos, estado):
            pesos = self.generador_azar().random(len(datos))
            return dict(zip(datos, pesos / pesos.sum(), strict=True))

    try:
        configs = [_config(estrategia="sorteo_prueba_simular", parametros={}, semilla=s) for s in (5, 6)]
        en_serie = [ejecutar_con_datos(c, datos).metricas for c in configs]
        barrera = threading.Barrier(2)

        def correr(config):
            barrera.wait()
            return ejecutar_con_datos(config, datos).metricas

        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(correr, configs)) == en_serie
        assert ejecutar_experimento(configs[0], FuenteSintetica()).metricas == en_serie[0]
        assert en_serie[0] != en_serie[1]
    finally:
        from ypyquant.estrategias import registro
        registro._REGISTRO.pop("sorteo_prueba_simular", None)
