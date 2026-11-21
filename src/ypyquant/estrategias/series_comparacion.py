"""
Series temporales de la comparación de experimentos: drawdown, Sharpe y volatilidad móviles y turnover por rebalanceo.

Responsabilidad:
- Funciones PURAS (listas/arrays y DataFrames, sin E/S) que calculan, a partir de la equity de cada experimento, las
  series que la página Comparar dibuja «en el tiempo» (`INDICADORES_TEMPORALES`). La equity base 100 ya la sirve
  `GET /api/comparar`; los indicadores escalares salen de `metricas`.
- `periodos_por_anio(fechas)`: periodicidad inferida de la mediana de huecos entre fechas (252 diaria, 52 semanal,
  12 mensual, 4 trimestral). Importa porque los experimentos importados del TFM son MENSUALES y los backtests
  de la interfaz DIARIOS: la ventana «12 meses» son 12 observaciones en un caso y 252 en el otro.
- `drawdown_serie`, `volatilidad_movil`, `sharpe_movil`: sobre una equity (cualquier escala). Definición igual a la de
  `PerformanceAnalyzer`/`_calcular_metricas` con rf = 0: Sharpe = retorno anualizado / volatilidad anualizada
  (con ventana de un año el retorno anualizado es el retorno de la ventana); volatilidad = desv. típica muestral de los
  retornos por periodo × √periodos_por_año. Antes de completar la ventana o con volatilidad nula el valor es `None`
  (sin dato), nunca 0.
- `turnover_por_rebalanceo(pesos)`: ½ Σ|w_t − w_(t−1)| entre pesos OBJETIVO consecutivos. Es una aproximación: el motor
  mide la rotación desde los pesos a la deriva (`turnover_anual` de las métricas), que no se guarda en `pesos.parquet`.

Cómo se alinean las frecuencias mixtas: `GET /api/comparar/series` calcula todo sobre el eje de fechas COMÚN que ya usa
la equity de `AlmacenExperimentos.comparar` (intersección exacta de fechas). Con un TFM mensual y un backtest diario el
eje es mensual y la ventana de 12 meses son 12 observaciones para todos; si todos son diarios, 252. El turnover no se
alinea: son los rebalanceos propios de cada experimento recortados al rango común.

Se relaciona con: app/api/rutas_experimentos.py (`GET /api/comparar/series`), ypyquant.estrategias.almacen (`comparar`,
`leer`) y app/web/src/lib/indicadores.ts (los mismos indicadores en la interfaz).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

# Indicadores «en el tiempo» que sirve el endpoint (clave -> etiqueta). La equity base 100 va por `GET /api/comparar`.
INDICADORES_TEMPORALES = {
    "drawdown": "Drawdown",
    "sharpe_movil": "Sharpe móvil 12 meses",
    "volatilidad_movil": "Volatilidad móvil 12 meses",
    "turnover_rebalanceo": "Turnover por rebalanceo",
}


def periodos_por_anio(fechas: pd.DatetimeIndex | Sequence[Any]) -> int:
    """Periodos por año inferidos de la mediana del hueco entre fechas (252 si no hay información suficiente)."""
    indice = pd.DatetimeIndex(pd.to_datetime(list(fechas))).sort_values()
    if len(indice) < 2:
        return 252
    hueco = float(np.median(np.diff(indice.values).astype("timedelta64[D]").astype(float)))
    if hueco <= 4:
        return 252
    if hueco <= 10:
        return 52
    if hueco <= 45:
        return 12
    return max(1, round(365.25 / hueco))


def _lista(valores: pd.Series | np.ndarray) -> list[float | None]:
    """Serie a lista JSON-segura: NaN e infinitos pasan a `None`."""
    return [float(v) if np.isfinite(v) else None for v in np.asarray(valores, dtype=float)]


def drawdown_serie(equity: Sequence[float]) -> list[float | None]:
    """Caída desde el máximo acumulado en cada punto (0 en máximos, negativa bajo ellos)."""
    serie = pd.Series(np.asarray(equity, dtype=float))
    if serie.empty:
        return []
    return _lista(serie / serie.cummax() - 1.0)


def volatilidad_movil(equity: Sequence[float], periodos_anio: int, ventana: int | None = None) -> list[float | None]:
    """Volatilidad anualizada de los retornos de las últimas `ventana` observaciones (por defecto un año)."""
    ventana = ventana or periodos_anio
    retornos = pd.Series(np.asarray(equity, dtype=float)).pct_change()
    return _lista(retornos.rolling(ventana).std(ddof=1) * np.sqrt(periodos_anio))


def sharpe_movil(equity: Sequence[float], periodos_anio: int, ventana: int | None = None) -> list[float | None]:
    """
    Sharpe (rf = 0) de las últimas `ventana` observaciones: retorno anualizado de la ventana / volatilidad anualizada.
    `None` si la ventana no está completa o la volatilidad es nula (Sharpe indefinido).
    """
    ventana = ventana or periodos_anio
    serie = pd.Series(np.asarray(equity, dtype=float))
    retornos = serie.pct_change()
    retorno_ventana = serie / serie.shift(ventana) - 1.0
    anualizado = (1.0 + retorno_ventana) ** (periodos_anio / ventana) - 1.0
    volatilidad = retornos.rolling(ventana).std(ddof=1) * np.sqrt(periodos_anio)
    return _lista(anualizado / volatilidad.where(volatilidad > 0))


def turnover_por_rebalanceo(pesos: pd.DataFrame) -> tuple[list[str], list[float]]:
    """
    `(fechas, turnover)` por rebalanceo: ½ Σ|w − w_anterior| entre pesos objetivo consecutivos (un activo que sale
    pasa a 0). El primer rebalanceo (compra inicial) no cuenta. `pesos` tiene columnas `fecha`, `ticker`, `peso`.
    """
    if pesos is None or pesos.empty:
        return [], []
    matriz = pesos.pivot_table(index="fecha", columns="ticker", values="peso", aggfunc="sum").sort_index().fillna(0.0)
    cambios = 0.5 * matriz.diff().abs().sum(axis=1).iloc[1:]
    return [pd.Timestamp(f).strftime("%Y-%m-%d") for f in cambios.index], [float(v) for v in cambios.values]
