"""
Evaluación: métricas, validaciones y procedencia de CUALQUIER `Simulacion`.

Responsabilidad:
- `evaluar(simulacion, benchmark=None, validar=True) -> Evaluacion`: a partir de los hechos de una
  simulación (equity, pesos, operaciones, rotaciones) calcula todas las métricas del experimento
  (retorno/riesgo con `PerformanceAnalyzer`, Ulcer, Martin, costes, turnover, tasa de acierto y alfa/beta
  frente al benchmark), añade el benchmark reescalado a la curva de equity y ejecuta los validadores
  registrados que pida `config.validaciones`. Es el módulo profundo detrás de una interfaz pequeña: no sabe
  de dónde viene la simulación (backtest hoy; paper interno o Alpaca mañana, construidas con
  `simulacion_desde_tablas` a partir de lo que realmente pasó).
- `Evaluacion` (dataclass): `metricas`, `equity` (con la columna `benchmark`) y `procedencia`.
- `desglose_por_motivo(operaciones)`: en estrategias de señales (operaciones con columna `motivo`), agrupa las SALIDAS por motivo
  y lado de la posición con nº, PnL, tasa de acierto y duración media; sale en `metricas['motivos_salida']` y sirve para ver
  qué regla (stop, take profit, trailing, tiempo, señal...) gana o pierde dinero.
- `reevaluar(config, equity, pesos, operaciones, ...)`: recalcula las métricas de un experimento ya guardado
  solo con sus tablas Parquet, sin repetir el backtest (la usa `AlmacenExperimentos.reevaluar` y el script
  `scripts/reevaluar_experimentos.py`). Si el experimento no guardó `rotaciones.parquet` (guardados antes de
  este módulo), el turnover se ESTIMA desde operaciones y equity (ver `estimar_rotaciones`) y se marca en
  `metricas['reevaluacion']`.

Se relaciona con:
- ypyquant.estrategias.simulacion (`Simulacion`, `simulacion_desde_tablas`) y .experimentos (`VALIDADORES`,
  `ResultadoExperimento`, `ejecutar_con_datos` = `simular` -> `evaluar`).
- ypyquant.portfolio.performance_metrics (`PerformanceAnalyzer`) y ypyquant.validacion (Ulcer, Martin).
- ypyquant.estrategias.almacen (persistencia) y scripts/reevaluar_experimentos.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ypyquant.estrategias.experimentos import (
    DIAS_ANIO,
    VALIDADORES,
    ConfiguracionExperimento,
    ResultadoExperimento,
    _limpiar,
)
from ypyquant.estrategias.senales_modelo import AJUSTE, LLAMADA_MARGEN, MOTIVOS, MOTIVOS_ENTRADA, a_motivo_actual
from ypyquant.estrategias.simulacion import Simulacion, simulacion_desde_tablas
from ypyquant.execution.ejecutor import TICKER_EFECTIVO
from ypyquant.portfolio.performance_metrics import PerformanceAnalyzer
from ypyquant.validacion import indice_ulcer, ratio_martin


@dataclass
class Evaluacion:
    """
    Resultado de evaluar una `Simulacion`.

    metricas: todas las métricas (JSON válido; `validacion` con la salida de los validadores pedidos).
    equity: la de la simulación más la columna `benchmark` (reescalado al capital inicial; NaN si no hay).
    procedencia: la de la simulación (commit, versiones, hash de datos), tal cual.
    """

    metricas: dict[str, Any]
    equity: pd.DataFrame
    procedencia: dict[str, Any] = field(default_factory=dict)


def _turnover(rotaciones: pd.Series, anios: float) -> tuple[float, float]:
    """
    (turnover medio por rebalanceo, turnover anual) con la rotación real del motor: lo operado
    desde los pesos a la deriva hasta el objetivo. La 1.ª (compra inicial) se excluye. Antes se
    comparaban pesos objetivo consecutivos, y una estrategia de pesos fijos (p. ej. pesos iguales)
    salía con turnover 0 aunque operase.
    """
    cambios = [float(r) for r in rotaciones.iloc[1:]]
    if not cambios:
        return 0.0, 0.0
    return float(np.mean(cambios)), float(np.sum(cambios) / anios) if anios > 0 else 0.0


def _exposiciones(pesos: pd.DataFrame) -> dict[str, float]:
    """
    Exposición de los pesos objetivo de cada rebalanceo (sin `CASH`): bruta (suma de |w|), neta (suma de w) y corta (suma de
    los negativos, en valor absoluto). Solo se calcula si la estrategia usó margen: algún peso corto o un `CASH` negativo
    (préstamo); si no, devuelve `{}` y las métricas de los experimentos largos de siempre no cambian.
    """
    if pesos.empty or not ((pesos["peso"] < 0).any()):
        return {}
    posiciones = pesos[pesos["ticker"] != TICKER_EFECTIVO]
    por_fecha = posiciones.groupby("fecha")["peso"]
    bruta = por_fecha.apply(lambda w: float(w.abs().sum()))
    neta = por_fecha.sum()
    corta = por_fecha.apply(lambda w: float(-w[w < 0].sum()))
    return {"exposicion_bruta_media": float(bruta.mean()), "exposicion_bruta_maxima": float(bruta.max()),
            "exposicion_neta_media": float(neta.mean()), "exposicion_corta_media": float(corta.mean())}


def _tasa_acierto(operaciones: pd.DataFrame) -> float:
    """
    Fracción de operaciones cerradas (con `pnl`) con pnl > 0; 0 si no hay ninguna (misma regla que el motor).
    Cierran una posición las ventas de un largo y las compras que cubren un corto.
    """
    cerradas = operaciones[operaciones["pnl"].notna()]
    return float((cerradas["pnl"] > 0).sum() / len(cerradas)) if len(cerradas) else 0.0


def _calcular_metricas(simulacion: Simulacion, equity: pd.DataFrame, benchmark_ret: pd.Series | None) -> dict[str, Any]:
    """
    Métricas del experimento: retorno/riesgo (PerformanceAnalyzer), Ulcer y Martin, costes, turnover y alfa/beta.

    `costes_sobre_ganancia_bruta` = comisiones_totales / (capital_final - capital_inicial + comisiones_totales);
    None si la ganancia bruta es <= 0. Un valor > 0,5 indica que los costes se comen la mitad del edge.
    """
    config = simulacion.config
    operaciones = simulacion.operaciones
    retornos = equity.set_index("fecha")["retorno"].dropna()
    base = PerformanceAnalyzer.calculate_metrics(retornos, risk_free_rate=0.0, periods_per_year=DIAS_ANIO)
    anios = max((pd.Timestamp(config.fecha_fin) - pd.Timestamp(config.fecha_inicio)).days / 365.25, 1e-9)
    turnover_medio, turnover_anual = _turnover(simulacion.rotaciones["rotacion"], anios)
    metricas: dict[str, Any] = {
        "retorno_total": base.total_return,
        "cagr": base.annualized_return,
        "volatilidad": base.volatility,
        "sharpe": base.sharpe_ratio,
        "sortino": base.sortino_ratio,
        "calmar": base.calmar_ratio,
        "max_drawdown": float(equity["drawdown"].min()),
        "var_95": base.var_95,
        "cvar_95": base.cvar_95,
        "capital_final": float(equity["equity"].iloc[-1]),
        "turnover_medio": turnover_medio,
        "turnover_anual": turnover_anual,
        "n_rebalanceos": int(simulacion.pesos["fecha"].nunique()),
        "n_operaciones": len(operaciones),
        "comisiones_totales": float(sum(operaciones["comision"].tolist())),  # suma secuencial, como el motor
        "tasa_acierto": _tasa_acierto(operaciones),
        "dias": len(equity),
        "ulcer": indice_ulcer(equity["equity"].to_numpy(dtype=float)),
        "martin": ratio_martin(retornos.to_numpy(dtype=float), periodos_anio=DIAS_ANIO),
    }
    # Costes sobre ganancia bruta = comisiones / (ganancia neta + comisiones). Por encima de 0,5 los costes
    # se comen la mitad del edge; sin ganancia bruta positiva no está definido (None).
    ganancia_bruta = metricas["capital_final"] - config.capital_inicial + metricas["comisiones_totales"]
    metricas["costes_sobre_ganancia_bruta"] = (
        metricas["comisiones_totales"] / ganancia_bruta if ganancia_bruta > 0 else float("nan"))
    if benchmark_ret is not None and len(benchmark_ret) > 1:
        comparacion = PerformanceAnalyzer.compare_to_benchmark(retornos, benchmark_ret, risk_free_rate=0.0)
        if "error" not in comparacion:
            metricas.update({
                "alfa": comparacion["alpha"], "beta": comparacion["beta"],
                "tracking_error": comparacion["tracking_error"],
                "information_ratio": comparacion["information_ratio"],
                "benchmark_retorno_total": comparacion["benchmark"].total_return,
                "benchmark_cagr": comparacion["benchmark"].annualized_return,
                "benchmark_sharpe": comparacion["benchmark"].sharpe_ratio,
            })
    metricas.update(_exposiciones(simulacion.pesos))  # margen: exposición bruta/neta (solo con cortos o préstamo)
    llamadas = sum(1 for aviso in simulacion.avisos if aviso.startswith("Llamada de margen"))
    if metricas.get("exposicion_bruta_maxima") is not None or llamadas:
        metricas["llamadas_margen"] = llamadas
    if "motivo" in operaciones.columns:  # estrategias de señales: qué regla de salida gana o pierde dinero
        metricas["motivos_salida"] = desglose_por_motivo(operaciones)
    metricas["validacion"] = {}  # los validadores registrados (CPCV/DSR/PBO...) escriben aquí
    return _limpiar(metricas)


def desglose_por_motivo(operaciones: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Desglose de las SALIDAS de una estrategia de señales por `(motivo, lado de la posición)`.

    Cada fila: `{motivo, lado, n, pnl_total, pnl_medio, tasa_acierto, duracion_media_dias}`, ordenadas por el orden de
    `MOTIVOS` y largo antes que corto. Una salida es una operación con `pnl` (venta que cierra un largo, compra que cubre un corto)
    cuyo motivo no es `ajuste` (venta parcial por deriva). El PnL es el de la operación de cierre, neto de su comisión pero NO de
    la de apertura. La duración (días naturales) va de la última entrada del ticker a la salida. Lee también el vocabulario
    anterior (`senal`, `stop_perdida`...). Sin columna `motivo` (estrategias sin señales) devuelve `[]`.
    """
    if "motivo" not in operaciones.columns or operaciones.empty:
        return []
    entradas: dict[str, pd.Timestamp] = {}
    salidas: dict[tuple[str, str], list[tuple[float, float | None]]] = {}
    for fecha, ticker, lado, pnl, motivo in zip(operaciones["fecha"], operaciones["ticker"], operaciones["lado"],
                                                operaciones["pnl"], operaciones["motivo"], strict=True):
        cierra = pd.notna(pnl)
        motivo = a_motivo_actual(motivo, es_entrada=not cierra)
        if motivo in MOTIVOS_ENTRADA:
            entradas[ticker] = pd.Timestamp(fecha)
        elif cierra and motivo != AJUSTE:
            entrada = entradas.get(ticker) if motivo == LLAMADA_MARGEN else entradas.pop(ticker, None)
            duracion = (pd.Timestamp(fecha) - entrada).days if entrada is not None else None
            salidas.setdefault((motivo, "largo" if lado == "venta" else "corto"), []).append((float(pnl), duracion))
    orden = {m: i for i, m in enumerate(MOTIVOS)}
    filas = []
    for (motivo, lado), registros in sorted(salidas.items(), key=lambda kv: (orden.get(kv[0][0], len(orden)), kv[0][1] != "largo")):
        pnls = [p for p, _ in registros]
        duraciones = [d for _, d in registros if d is not None]
        filas.append({"motivo": motivo, "lado": lado, "n": len(registros), "pnl_total": float(sum(pnls)),
                      "pnl_medio": float(sum(pnls) / len(pnls)), "tasa_acierto": sum(p > 0 for p in pnls) / len(pnls),
                      "duracion_media_dias": float(sum(duraciones) / len(duraciones)) if duraciones else None})
    return filas


def evaluar(simulacion: Simulacion, benchmark: pd.Series | None = None, validar: bool = True) -> Evaluacion:
    """
    Calcula métricas, curva de benchmark y validaciones de una `Simulacion`.

    Args:
        benchmark: precios (niveles) del benchmark; None usa `simulacion.benchmark_precios`. Se alinean con la
            equity (ffill y bfill) y se reescalan al capital inicial para la columna `benchmark`.
        validar: si False (permutaciones: miles de evaluaciones) no se ejecutan los validadores de
            `config.validaciones`. Lanza `ValueError` si piden uno no registrado.
    """
    config = simulacion.config
    equity = simulacion.equity.copy()
    precios_benchmark = simulacion.benchmark_precios if benchmark is None else benchmark
    benchmark_ret = None
    if precios_benchmark is not None:
        alineada = precios_benchmark.reindex(pd.DatetimeIndex(equity["fecha"])).ffill()
        if alineada.notna().any():
            alineada = alineada.bfill()
            equity["benchmark"] = (alineada / alineada.iloc[0] * config.capital_inicial).values
            benchmark_ret = alineada.pct_change().dropna()
    if "benchmark" not in equity:
        equity["benchmark"] = np.nan
    metricas = _calcular_metricas(simulacion, equity, benchmark_ret)
    if validar:
        resultado = ResultadoExperimento(config, metricas, equity, simulacion.pesos, simulacion.operaciones,
                                         list(simulacion.avisos))
        for nombre in config.validaciones:
            if nombre not in VALIDADORES:
                raise ValueError(f"Validador '{nombre}' no registrado. Disponibles: {sorted(VALIDADORES)}")
            metricas["validacion"][nombre] = _limpiar(VALIDADORES[nombre](resultado))
    return Evaluacion(metricas, equity, dict(simulacion.procedencia))


def reevaluar(config: ConfiguracionExperimento, equity: pd.DataFrame, pesos: pd.DataFrame, operaciones: pd.DataFrame,
              rotaciones: pd.DataFrame | None = None, avisos: list[str] | None = None) -> Evaluacion:
    """
    Recalcula las métricas de un experimento guardado a partir de sus tablas, sin repetir el backtest.

    `equity` es la guardada (con su columna `benchmark` ya reescalada: de ahí salen alfa y beta). Sin
    `rotaciones` el turnover se estima y `metricas['reevaluacion']` lo indica. No toca la procedencia
    (devuelve un dict vacío: el almacén conserva la original).
    """
    simulacion = simulacion_desde_tablas(config, equity, pesos, operaciones, rotaciones, avisos)
    evaluacion = evaluar(simulacion)
    if rotaciones is None:
        evaluacion.metricas["reevaluacion"] = {"turnover": "estimado desde operaciones (sin rotaciones.parquet)"}
    return evaluacion
