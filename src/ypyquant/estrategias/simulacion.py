"""
Simulación: qué pasó en una ejecución de una estrategia (sin métricas).

Responsabilidad:
- `Simulacion` (dataclass): la INTERFAZ común entre quien ejecuta y quien evalúa. Contiene solo hechos:
  `equity` (`fecha, equity, drawdown, retorno`), `pesos` (objetivos por rebalanceo, formato largo),
  `operaciones` (lo ejecutado), `rotaciones` (`fecha, rotacion`: lo operado en cada rebalanceo respecto a
  los pesos a la deriva), `avisos` y, opcionalmente, `benchmark_precios` y `procedencia` de los datos.
  Nada de Sharpe ni drawdown máximo: eso lo calcula `evaluar` (ver `evaluacion.py`).
- `simular(config, datos, rng=None, completo=True)`: corre la estrategia con el motor (`BacktestEngine`, que
  planifica con el Ejecutor) sobre precios ya cargados y devuelve una `Simulacion`. Aplica el universo
  (incluido el point-in-time `sp500_pit`), recorta a `<= fecha_fin` y avisa de los deslistados liquidados a
  precio caducado. Es la costura (seam): hoy el adaptador es el backtest; mañana el paper interno o Alpaca
  construirán `Simulacion` a partir de lo que realmente ocurrió y reutilizarán `evaluar`.
- Aleatoriedad: cada ejecución usa un generador PROPIO (`rng`), inyectado en la estrategia
  (`Estrategia.rng`); no se toca el estado global de `random` ni de `np.random`, así dos simulaciones en
  hilos distintos no interfieren. Si no se pasa `rng` se crea `np.random.default_rng(config.semilla)`, el
  mismo flujo que usaban las estrategias con la semilla (Michaud) antes del refactor.

Se relaciona con:
- ypyquant.estrategias.experimentos (`ConfiguracionExperimento`, `DatosExperimento`; `ejecutar_con_datos`
  compone `simular` -> `evaluar`) y .evaluacion (consume la `Simulacion`).
- ypyquant.estrategias.base (`Estrategia`, adaptador al motor) y .procedencia (hash de los datos).
- ypyquant.portfolio.backtest_engine (motor) y ypyquant.execution.ejecutor (plan de órdenes).
- ypyquant.estrategias.universo_dinamico (filtro point-in-time) y ypyquant.data.universos (membresías, cobertura).
- ypyquant.data.fuente_precios: los deslistados (fecha y motivo) los aporta la fuente vía `DatosExperimento.deslistados`;
  sin ellos se infieren de los precios, como siempre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ypyquant.data import universos
from ypyquant.data.fuente_precios import DIAS_PRECIO_CADUCADO, Deslistado, inferir_deslistado
from ypyquant.data.matriz_precios import extraer_serie_precios
from ypyquant.data.universos import aviso_cobertura_pit, es_universo_pit, resolver_universo
from ypyquant.estrategias.base import recortar_hasta
from ypyquant.estrategias.experimentos import ConfiguracionExperimento, DatosExperimento
from ypyquant.estrategias.procedencia import obtener_procedencia
from ypyquant.estrategias.registro import obtener_estrategia
from ypyquant.estrategias.universo_dinamico import UniversoPuntoEnElTiempo
from ypyquant.execution.margen import reglas_desde_parametros
from ypyquant.portfolio.backtest_engine import BacktestConfig, BacktestEngine

COLUMNAS_ROTACIONES = ["fecha", "rotacion"]


@dataclass
class Simulacion:
    """
    Hechos de una ejecución (backtest, paper interno, Alpaca...), sin ninguna métrica.

    Campos:
        config: configuración con la que se ejecutó (capital inicial, fechas, benchmark, validaciones).
        equity: `fecha, equity, drawdown, retorno` (el drawdown es la fracción bajo el máximo previo).
        pesos: `fecha, ticker, peso` (pesos objetivo en cada rebalanceo).
        operaciones: `fecha, ticker, lado, cantidad, precio, comision, pnl` (+ `motivo` en estrategias de señales).
        rotaciones: `fecha, rotacion` de cada rebalanceo (valor operado / 2·equity); la 1.ª es la compra inicial.
        avisos: mensajes no fatales de la ejecución.
        benchmark_precios: cierres del benchmark (None si no hay); `evaluar` lo alinea con la equity.
        procedencia: commit, versiones y hash de los datos (vacío si se pidió `completo=False`).
    """

    config: ConfiguracionExperimento
    equity: pd.DataFrame
    pesos: pd.DataFrame
    operaciones: pd.DataFrame
    rotaciones: pd.DataFrame
    avisos: list[str] = field(default_factory=list)
    benchmark_precios: pd.Series | None = None
    procedencia: dict[str, Any] = field(default_factory=dict)


def _motivos_de_operaciones(operaciones: pd.DataFrame, eventos: list[dict[str, Any]]) -> list[str]:
    """
    Motivo de cada operación de un backtest de una `EstrategiaSenales`, cruzando `(fecha, ticker, lado)` con sus eventos
    (`senal`, `stop_perdida`, `stop_trailing`, `take_profit`, `tiempo`, `fuera_universo`, `regla_propia`). Una operación
    sin evento (compra o venta parcial que solo recoloca los pesos por la deriva o los costes) es `ajuste`.
    """
    por_clave = {(e["fecha"], e["ticker"], "compra" if e["tipo"] == "entrada" else "venta"): e["motivo"] for e in eventos}
    return [por_clave.get((f"{fecha:%Y-%m-%d}", ticker, lado), "ajuste")
            for fecha, ticker, lado in zip(operaciones["fecha"], operaciones["ticker"], operaciones["lado"], strict=True)]


def _deslistados_de_la_simulacion(datos: DatosExperimento, precios: dict[str, pd.DataFrame],
                                  fin: pd.Timestamp) -> dict[str, Deslistado]:
    """
    Deslistados a tener en cuenta: los que trae la carga (explícitos de la fuente o inferidos por su adaptador).

    Si `datos` se construyó sin fuente (`deslistados is None`) se infieren de los precios: el último dato queda
    más de `DIAS_PRECIO_CADUCADO` antes de `fin`.
    """
    if datos.deslistados is not None:
        return datos.deslistados
    inferidos = {t: inferir_deslistado(df.index[-1], fin) for t, df in precios.items()}
    return {t: d for t, d in inferidos.items() if d is not None}


def _avisos_liquidaciones_caducadas(trades: list[Any], precios: dict[str, pd.DataFrame],
                                    deslistados: dict[str, Deslistado]) -> list[str]:
    """
    Avisos de posiciones vendidas con un precio caducado (el valor dejó de cotizar antes de la venta).

    El motor valora una posición deslistada a su último precio conocido y la liquida a ese precio en el
    siguiente rebalanceo; si el deslistamiento fue una quiebra el resultado real habría sido peor. La fecha y
    el motivo los aporta la fuente de precios (`deslistados`).
    """
    avisos = []
    vistos: set[str] = set()
    for trade in trades:
        if trade.side != "sell" or trade.ticker in vistos or trade.ticker not in precios:
            continue
        deslistado = deslistados.get(trade.ticker)
        if deslistado is None:
            continue
        if pd.Timestamp(trade.entry_date) - deslistado.fecha > pd.Timedelta(days=DIAS_PRECIO_CADUCADO):
            vistos.add(trade.ticker)
            motivo = f" ({deslistado.motivo})" if deslistado.motivo else ""
            avisos.append(f"{trade.ticker} se liquidó a su último precio conocido ({deslistado.fecha:%Y-%m-%d}) el "
                          f"{pd.Timestamp(trade.entry_date):%Y-%m-%d}: dejó de cotizar (deslistado{motivo}) con la posición abierta.")
    return avisos


def _benchmark_de_datos(config: ConfiguracionExperimento, datos: DatosExperimento, fin: pd.Timestamp,
                        ) -> tuple[pd.DataFrame | None, list[str]]:
    """OHLCV del benchmark recortado a `<= fin` (None si no hay datos de ese ticker) y los avisos correspondientes."""
    if datos.ticker_benchmark != config.benchmark:
        return None, [f"Sin datos del benchmark {config.benchmark} (los cargados son de {datos.ticker_benchmark})"]
    if datos.benchmark is None:
        return None, [f"Sin datos del benchmark {config.benchmark}"]
    return recortar_hasta({config.benchmark: datos.benchmark}, fin).get(config.benchmark), []


def simular(config: ConfiguracionExperimento, datos: DatosExperimento, rng: np.random.Generator | None = None,
            completo: bool = True) -> Simulacion:
    """
    Corre la estrategia de `config` con el motor sobre precios ya cargados y devuelve la `Simulacion` (sin métricas).

    Pasos: normaliza la configuración, inyecta semilla y generador propios en la estrategia, selecciona de
    `datos.precios` los tickers del universo y recorta todo a `<= fecha_fin` (así un barrido cargado hasta una
    fecha posterior no deja ver el futuro a `Estrategia.preparar`) y simula con `BacktestEngine`. Lanza
    `RuntimeError` si no hay precios o la estrategia no produjo ningún rebalanceo válido.

    Args:
        rng: generador de esta ejecución (None: `default_rng(config.semilla)`). Se pasa explícitamente a la
            estrategia; no se modifica ningún estado global de azar.
        completo: si False (tests de permutación: miles de backtests) se omite la procedencia (hash de precios, git).
    """
    config = config.normalizada()
    clase = obtener_estrategia(config.estrategia)
    estrategia = clase(config.parametros)
    estrategia.semilla = config.semilla
    estrategia.rng = rng if rng is not None else np.random.default_rng(config.semilla)

    tickers = resolver_universo(config.universo, inicio=config.fecha_inicio, fin=config.fecha_fin)
    fin = pd.Timestamp(config.fecha_fin)
    datos_universo = {t: datos.precios[t] for t in tickers if t in datos.precios}
    if not datos_universo:
        raise RuntimeError("No hay precios para ningún ticker del universo (¿sin red y sin caché?)")
    precios = recortar_hasta(datos_universo, fin)
    pit = es_universo_pit(config.universo)
    if pit:
        avisos = [aviso_cobertura_pit(config.fecha_inicio, config.fecha_fin, set(precios))]
        # La estrategia solo ve, en cada fecha, los miembros del índice en esa fecha con historia hasta t-1
        filtro_universo = UniversoPuntoEnElTiempo(universos.cargar_membresias_sp500()).crear_filtro(precios)  # membresías: solo desde `universos` (una costura)
    else:
        avisos = [f"Sin datos para {t}" for t in tickers if t not in precios]
        filtro_universo = None

    # Margen y cortos (reglas compartidas con el paper interno, `ypyquant.execution.margen`): apalancamiento (préstamo a
    # tasa libre + diferencial) y/o posiciones cortas (`exposicion_corta`, con coste de préstamo de títulos). Sin esos
    # parámetros las reglas están inactivas y el motor funciona como siempre.
    reglas = reglas_desde_parametros(estrategia.parametros)
    motor = BacktestEngine(BacktestConfig(
        start_date=config.fecha_inicio, end_date=config.fecha_fin, initial_capital=config.capital_inicial,
        rebalance_frequency=config.frecuencia_rebalanceo, commission_rate=config.comision,
        slippage_pct=config.slippage, max_positions=max(len(precios), 1), benchmark_ticker=config.benchmark,
    ), reglas_margen=reglas)
    resultado_bt = motor.run_backtest(estrategia.como_funcion(filtro_universo=filtro_universo), precios)
    if not estrategia.historial_pesos:
        detalle = estrategia.errores[0] if estrategia.errores else "sin pesos"
        raise RuntimeError(f"La estrategia no generó ningún rebalanceo válido ({detalle})")
    avisos.extend(estrategia.avisos)
    if reglas.activo:
        avisos.extend(avisos_de_margen(motor, reglas, config.capital_inicial))
    if estrategia.errores:
        avisos.append(f"{len(estrategia.errores)} rebalanceos fallaron; primero: {estrategia.errores[0]}")
    if pit:
        avisos.extend(_avisos_liquidaciones_caducadas(motor.trades, precios, _deslistados_de_la_simulacion(datos, precios, fin)))
    benchmark_usado, avisos_benchmark = _benchmark_de_datos(config, datos, fin)
    avisos.extend(avisos_benchmark)

    curva = resultado_bt.equity_curve
    equity = pd.DataFrame({"fecha": curva.index, "equity": curva["equity"].values,
                           "drawdown": curva["drawdown_pct"].values, "retorno": curva["returns"].values})
    pesos = pd.DataFrame(
        [(pd.Timestamp(f), t, w) for f, pesos_f in estrategia.historial_pesos for t, w in pesos_f.items()],
        columns=["fecha", "ticker", "peso"],
    )
    operaciones = pd.DataFrame([{
        "fecha": pd.Timestamp(t.entry_date), "ticker": t.ticker, "lado": "compra" if t.side == "buy" else "venta",
        "cantidad": t.quantity, "precio": t.exit_price if t.exit_price is not None else t.entry_price,
        "comision": t.commission, "pnl": t.pnl,
    } for t in motor.trades], columns=["fecha", "ticker", "lado", "cantidad", "precio", "comision", "pnl"])
    if getattr(estrategia, "eventos", None):  # estrategias de señales: cada operación lleva su motivo
        operaciones["motivo"] = _motivos_de_operaciones(operaciones, estrategia.eventos)
    rotaciones = pd.DataFrame([(pd.Timestamp(f), float(r)) for f, r in resultado_bt.rotaciones],
                              columns=COLUMNAS_ROTACIONES)
    procedencia = obtener_procedencia(precios, benchmark_usado) if completo else {}
    return Simulacion(
        config, equity, pesos, operaciones, rotaciones, avisos,
        benchmark_precios=extraer_serie_precios(benchmark_usado, config.benchmark, avisar=False),
        procedencia=procedencia,
    )


def avisos_de_margen(motor: BacktestEngine, reglas, capital_inicial: float) -> list[str]:
    """
    Avisos del margen de una ejecución (apalancada y/o con cortos): intereses del préstamo, préstamo de títulos,
    exposición bruta máxima y UNA línea por llamada de margen (fecha, equity, requisito y fracción conservada).
    """
    avisos = []
    if (reglas.apalancamiento_maximo > 1.0 and not reglas.permite_cortos) or motor.coste_financiacion > 0:
        avisos.append(f"Apalancamiento: intereses del préstamo de margen {motor.coste_financiacion:,.0f} "
                      f"({motor.coste_financiacion / capital_inicial:.1%} del capital inicial) a "
                      f"{reglas.tasa_financiacion:.2%} anual; exposición bruta máxima observada {motor.exposicion_maxima:.2f}x.")
    if reglas.permite_cortos:
        avisos.append(f"Cortos: préstamo de títulos {motor.coste_prestamo_titulos:,.0f} "
                      f"({motor.coste_prestamo_titulos / capital_inicial:.1%} del capital inicial) a "
                      f"{reglas.coste_prestamo_titulos:.2%} anual; exposición bruta máxima observada {motor.exposicion_maxima:.2f}x.")
    for llamada in motor.llamadas_margen:
        avisos.append(f"Llamada de margen {pd.Timestamp(llamada['fecha']):%Y-%m-%d}: equity {llamada['equity']:,.0f} por debajo del "
                      f"mantenimiento {llamada['requisito_mantenimiento']:,.0f} (exposición bruta {llamada['bruta']:,.0f}); "
                      f"liquidación forzosa: se conserva el {llamada['factor_conservado']:.0%} de las posiciones.")
    return avisos


def estimar_rotaciones(equity: pd.DataFrame, pesos: pd.DataFrame, operaciones: pd.DataFrame) -> pd.DataFrame:
    """
    Rotación aproximada de cada rebalanceo cuando no se guardó la real: valor operado ese día / (2·equity).

    Las fechas de rebalanceo son las de `pesos`; los rebalanceos sin operaciones rotan 0. Es una APROXIMACIÓN:
    usa los precios de ejecución (con slippage) y deja fuera lo que el Ejecutor planificó pero no operó por
    estar bajo el umbral, así que el turnover reevaluado puede diferir un poco del original.
    """
    fechas = sorted(pd.to_datetime(pesos["fecha"]).unique())
    valor = (operaciones["cantidad"] * operaciones["precio"]).groupby(pd.to_datetime(operaciones["fecha"])).sum()
    serie_equity = equity.set_index(pd.to_datetime(equity["fecha"]))["equity"]
    filas = []
    for fecha in fechas:
        base = serie_equity.asof(fecha)
        operado = float(valor.get(fecha, 0.0))
        filas.append((pd.Timestamp(fecha), operado / (2 * base) if base and base > 0 else 0.0))
    return pd.DataFrame(filas, columns=COLUMNAS_ROTACIONES)


def simulacion_desde_tablas(config: ConfiguracionExperimento, equity: pd.DataFrame, pesos: pd.DataFrame,
                            operaciones: pd.DataFrame, rotaciones: pd.DataFrame | None = None,
                            avisos: list[str] | None = None, procedencia: dict[str, Any] | None = None,
                            benchmark_precios: pd.Series | None = None) -> Simulacion:
    """
    Construye una `Simulacion` desde tablas ya existentes: un experimento guardado (Parquet) o, mañana, lo que
    realmente ocurrió en el paper interno o en Alpaca. Es el adaptador de entrada de `evaluar`.

    Si la `equity` trae una columna `benchmark` con datos (la guardada, ya reescalada) se usa como precios del
    benchmark (salvo que se pase `benchmark_precios`). Sin `rotaciones` se estiman con `estimar_rotaciones`.
    """
    benchmark = benchmark_precios
    if benchmark is None and "benchmark" in equity.columns and equity["benchmark"].notna().any():
        benchmark = equity.set_index(pd.to_datetime(equity["fecha"]))["benchmark"]
    equity = equity[[c for c in equity.columns if c != "benchmark"]].copy()
    return Simulacion(config, equity, pesos, operaciones,
                      rotaciones if rotaciones is not None else estimar_rotaciones(equity, pesos, operaciones),
                      list(avisos or []), benchmark_precios=benchmark, procedencia=dict(procedencia or {}))
