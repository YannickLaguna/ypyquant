"""
Tests de las decisiones del 2026-11-22 sobre optimización: volatilidad objetivo inalcanzable (aviso explícito en la
fachada y en los avisos del experimento) y apalancamiento como parámetro de las estrategias con volatilidad objetivo.

Se relaciona con: ypyquant.optimization.optimizador_portafolio (fachada), ypyquant.estrategias.incluidas
(estrategias), ypyquant.estrategias.base (`sanear_pesos`), ypyquant.portfolio.backtest_engine (margen).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ypyquant.data.fuentes import FuenteSintetica
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, ejecutar_experimento
from ypyquant.optimization import optimizador_portafolio as fachada

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
ESTRATEGIAS_CON_VOLATILIDAD = ["bayes_stein", "bl_prior", "michaud", "robusta_elipsoide", "robusta_caja"]


def _config(**cambios) -> ConfiguracionExperimento:
    base = {"estrategia": "bayes_stein", "universo": TICKERS, "fecha_inicio": "2020-01-01", "fecha_fin": "2021-12-31"}
    return ConfiguracionExperimento(**{**base, **cambios})


def _parametros(estrategia: str, **extra) -> dict:
    return {**({"n_replicas": 6} if estrategia == "michaud" else {}), **extra}


def _retornos(semilla: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    return pd.DataFrame(rng.normal(0.0004, 0.01, size=(500, 4)), columns=list("ABCD"))


# ---- tarea 1: volatilidad objetivo inalcanzable -> mínima varianza, y se dice ---------------------


def test_fachada_vol_objetivo_inalcanzable_trae_aviso_legible():
    res = fachada.optimizar_volatilidad_objetivo(_retornos(), 0.001)
    assert res["estado"] == "gmv_volatilidad_inalcanzable" and res["volatility_achieved"] is False
    esperado = f"σ objetivo 0,1 % < σ mínima alcanzable {res['volatility'] * 100:.1f}".replace(".", ",")
    assert res["aviso"].startswith(esperado) and res["aviso"].endswith("se usa la cartera de mínima varianza")


def test_fachada_vol_objetivo_alcanzable_no_trae_aviso():
    res = fachada.optimizar_volatilidad_objetivo(_retornos(), 0.15)
    assert res["aviso"] is None


@pytest.mark.parametrize("estrategia", ESTRATEGIAS_CON_VOLATILIDAD)
def test_experimento_avisa_agregado_cuando_la_vol_objetivo_es_inalcanzable(estrategia):
    resultado = ejecutar_experimento(
        _config(estrategia=estrategia, parametros=_parametros(estrategia, volatilidad_objetivo=0.001)),
        FuenteSintetica())
    avisos = [a for a in resultado.avisos if "mínima varianza" in a]
    assert len(avisos) == 1, resultado.avisos  # agregado: uno solo aunque haya muchos rebalanceos
    n_rebalanceos = resultado.metricas["n_rebalanceos"]
    assert n_rebalanceos > 3
    assert "σ objetivo 0,1 %" in avisos[0] and "σ mínima alcanzable" in avisos[0]
    assert f"de {n_rebalanceos} rebalanceos" in avisos[0] or "réplicas" in avisos[0]
    assert "2020-" in avisos[0] or "2021-" in avisos[0]  # fecha de rebalanceo (primero/último)


@pytest.mark.parametrize("estrategia", ESTRATEGIAS_CON_VOLATILIDAD)
def test_experimento_sin_vol_inalcanzable_no_avisa(estrategia):
    resultado = ejecutar_experimento(
        _config(estrategia=estrategia, parametros=_parametros(estrategia, volatilidad_objetivo=0.9)), FuenteSintetica())
    assert not [a for a in resultado.avisos if "mínima varianza" in a]


def test_el_aviso_llega_al_detalle_guardado_del_experimento(tmp_path):
    """Camino de la API/web: el aviso agregado se guarda en `metricas.json` y el detalle lo devuelve en `metricas_completas`."""
    from ypyquant.estrategias.almacen import AlmacenExperimentos

    almacen = AlmacenExperimentos(tmp_path)
    config = _config(parametros={"volatilidad_objetivo": 0.001}).normalizada()
    id_experimento = almacen.registrar(config)
    almacen.guardar(id_experimento, ejecutar_experimento(config, FuenteSintetica()))
    avisos = almacen.leer(id_experimento)["metricas_completas"]["avisos"]
    assert any("Volatilidad objetivo inalcanzable" in a and "mínima varianza" in a for a in avisos)


# ---- tarea 2: apalancamiento ---------------------------------------------------------------------


def test_sanear_pesos_conserva_el_efectivo_negativo_y_no_cambia_sin_el():
    from ypyquant.estrategias.base import sanear_pesos

    assert sanear_pesos({"A": 0.5, "B": 0.5}) == {"A": 0.5, "B": 0.5}  # sin cambios
    assert sanear_pesos({"A": 3.0, "B": 1.0}) == {"A": 0.75, "B": 0.25}  # long-only normaliza a 1, como siempre
    saneado = sanear_pesos({"A": 1.0, "B": 1.0, "CASH": -0.5})  # préstamo del 50 %: posiciones suman 1,5
    assert saneado == pytest.approx({"A": 0.75, "B": 0.75, "CASH": -0.5})
    assert sum(saneado.values()) == pytest.approx(1.0)
    assert sanear_pesos({"A": 1.0, "CASH": float("nan")}) == {"A": 1.0}


def _precios_lineales(n=120, crece=0.001) -> dict[str, pd.DataFrame]:
    indice = pd.bdate_range("2021-01-04", periods=n)
    cierre = pd.Series(100.0 * (1 + crece) ** np.arange(n), index=indice)
    return {"AAA": pd.DataFrame({"Close": cierre, "adjusted_close": cierre})}


def _motor(**extra):
    from ypyquant.portfolio.backtest_engine import BacktestConfig, BacktestEngine

    config = BacktestConfig(start_date="2021-01-04", end_date="2021-06-30", initial_capital=100_000.0,
                            rebalance_frequency="monthly", commission_rate=0.0, slippage_pct=0.0)
    return BacktestEngine(config, **extra)


def test_motor_con_cash_negativo_toma_prestamo_cobra_intereses_y_amplifica_el_retorno():
    precios = _precios_lineales()
    estrategia = lambda fecha, datos: {"AAA": 1.5, "CASH": -0.5}  # noqa: E731
    sin = _motor().run_backtest(lambda f, d: {"AAA": 1.0}, precios)
    con_motor = _motor(apalancamiento_maximo=1.5, tasa_financiacion=0.05)
    con = con_motor.run_backtest(estrategia, precios)
    assert con_motor.exposicion_maxima == pytest.approx(1.5, rel=1e-3)
    assert con_motor.coste_financiacion > 0
    assert any(fila["cash"] < 0 for fila in con_motor.equity_curve)
    assert con.total_return_pct > sin.total_return_pct  # el activo sube más que el coste del préstamo
    gratis = _motor(apalancamiento_maximo=1.5, tasa_financiacion=0.0)
    assert gratis.run_backtest(estrategia, precios).total_return_pct > con.total_return_pct


def test_motor_sin_apalancamiento_ignora_cash_negativo_como_siempre():
    """Con L = 1 (defecto) la política normaliza y no hay préstamo: el comportamiento histórico no cambia."""
    precios = _precios_lineales()
    historico = _motor()
    historico.run_backtest(lambda f, d: {"AAA": 1.5, "CASH": -0.5}, precios)
    assert all(fila["cash"] >= -1e-6 for fila in historico.equity_curve)
    assert historico.coste_financiacion == 0.0


def test_motor_rechaza_apalancamiento_menor_que_uno():
    with pytest.raises(ValueError, match="apalancamiento_maximo"):
        _motor(apalancamiento_maximo=0.5)


def test_apalancamiento_exige_volatilidad_objetivo_y_respeta_limites():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="volatilidad_objetivo"):
        _config(parametros={"apalancamiento_maximo": 1.5}).normalizada()
    with pytest.raises(ValidationError):
        _config(parametros={"apalancamiento_maximo": 0.8, "volatilidad_objetivo": 0.2}).normalizada()
    with pytest.raises(ValidationError):
        _config(parametros={"apalancamiento_maximo": 3.0, "volatilidad_objetivo": 0.2}).normalizada()
    _config(parametros={"apalancamiento_maximo": 1.5, "volatilidad_objetivo": 0.2}).normalizada()


@pytest.mark.parametrize("estrategia", ["bayes_stein", "bl_prior", "michaud", "robusta_elipsoide", "robusta_caja"])
def test_el_defecto_del_apalancamiento_no_cambia_la_huella(estrategia):
    base = _config(estrategia=estrategia, parametros=_parametros(estrategia, volatilidad_objetivo=0.2))
    explicita = _config(estrategia=estrategia, parametros=_parametros(
        estrategia, volatilidad_objetivo=0.2, apalancamiento_maximo=1.0, diferencial_financiacion=0.01))
    assert base.normalizada().identificador() == explicita.normalizada().identificador()
    con = _config(estrategia=estrategia, parametros=_parametros(
        estrategia, volatilidad_objetivo=0.2, apalancamiento_maximo=1.5))
    assert con.normalizada().identificador() != base.normalizada().identificador()


def test_el_formulario_de_la_web_recibe_los_parametros_de_apalancamiento():
    from ypyquant.estrategias.registro import obtener_estrategia

    for nombre in ("bayes_stein", "bl_prior", "michaud", "robusta_elipsoide", "robusta_caja"):
        propiedades = obtener_estrategia(nombre).Parametros.model_json_schema()["properties"]
        assert propiedades["apalancamiento_maximo"]["default"] == 1.0 and propiedades["apalancamiento_maximo"]["minimum"] == 1.0
        assert "diferencial_financiacion" in propiedades
    for nombre in ("pesos_iguales", "momentum", "preset_ypyquant"):  # no usan volatilidad objetivo: sin apalancamiento
        assert "apalancamiento_maximo" not in obtener_estrategia(nombre).Parametros.model_json_schema()["properties"]


@pytest.mark.parametrize("estrategia", ["bayes_stein", "robusta_caja", "michaud"])
def test_experimento_apalancado_refleja_exposicion_bruta_mayor_que_uno(estrategia):
    """σ* alta e inalcanzable sin apalancar: la cartera se escala, el préstamo llega al motor y se avisa."""
    sin = ejecutar_experimento(_config(estrategia=estrategia, parametros=_parametros(
        estrategia, volatilidad_objetivo=0.6)), FuenteSintetica())
    con = ejecutar_experimento(_config(estrategia=estrategia, parametros=_parametros(
        estrategia, volatilidad_objetivo=0.6, apalancamiento_maximo=2.0)), FuenteSintetica())
    suma = con.pesos.groupby("fecha")["peso"].sum()
    largas = con.pesos[con.pesos["ticker"] != "CASH"].groupby("fecha")["peso"].sum()
    assert suma.round(6).eq(1.0).all()  # posiciones + préstamo (CASH negativo) = 1
    assert largas.max() > 1.0 + 1e-6 and largas.max() <= 2.0 + 1e-9
    assert (con.pesos[con.pesos["ticker"] == "CASH"]["peso"] < 0).all()
    assert not (sin.pesos["ticker"] == "CASH").any()
    assert any(a.startswith("Apalancamiento (máximo 2.00x)") for a in con.avisos)
    assert any(a.startswith("Apalancamiento: intereses") for a in con.avisos)
