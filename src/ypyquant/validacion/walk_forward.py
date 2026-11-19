"""
Selección walk-forward sobre una matriz de retornos de un barrido de configuraciones.

Responsabilidad:
- `seleccion_walk_forward`: en cada fecha de reoptimización elige la configuración con mejor
  métrica en la ventana de entrenamiento ANTERIOR y concatena sus retornos fuera de muestra.
  Responde «¿qué habría obtenido alguien que re-optimiza periódicamente con esta rejilla?».

Aproximación (importante): cada columna es un backtest completo e independiente de toda la
muestra, así que se ignora (a) el coste de cambiar de configuración al cambiar de elección
(turnover entre estrategias) y (b) el estado de la estrategia (posiciones abiertas, indicadores
con calentamiento) al empezar a usar una columna en t. Es optimista en costes y aproxima el
estado con el de un backtest que ya llevaba funcionando.

Se relaciona con:
- ypyquant.validacion.estadisticos (`sharpe`, métrica por defecto).
- ypyquant.estrategias.experimentos (los barridos producen la matriz de retornos de entrada).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ypyquant.validacion.estadisticos import Metrica, sharpe


def seleccion_walk_forward(
    matriz_retornos: pd.DataFrame,
    ventana_entreno: int,
    paso: int,
    metrica: Metrica = sharpe,
    minimo_observaciones: int | None = None,
) -> tuple[pd.Series, pd.DataFrame, float]:
    """
    Walk-forward con selección de la mejor columna por métrica.

    Para t = ventana_entreno, ventana_entreno + paso, ... (posiciones de fila): elige la columna
    con mayor métrica en las filas [t − ventana_entreno, t) (sin incluir t, sin mirar el futuro) y
    usa sus retornos en [t, t + paso) (el último tramo puede ser más corto).

    Args:
        matriz_retornos: DataFrame T×N (índice=fechas, columnas=configuraciones). Los NaN se
            ignoran al calcular la métrica.
        ventana_entreno: filas de entrenamiento (> 0).
        paso: filas entre reoptimizaciones (> 0).
        metrica: callable(ndarray 2D) -> ndarray por columna; mayor es mejor. Por defecto Sharpe.
        minimo_observaciones: observaciones no NaN mínimas en la ventana para que una columna
            sea elegible; None = `ventana_entreno` (ventana completa). Si ninguna lo es, ese
            tramo queda en NaN y la elección en NaN.

    Returns:
        (retornos OOS concatenados [Series desde la fila `ventana_entreno`],
         elecciones [DataFrame índice=fecha de reoptimización, columnas `columna` y `metrica_entreno`],
         métrica de la serie OOS, calculada con la misma `metrica`).
    """
    if ventana_entreno <= 0 or paso <= 0:
        raise ValueError("ventana_entreno y paso deben ser positivos")
    if len(matriz_retornos) <= ventana_entreno:
        raise ValueError("La matriz no tiene más filas que la ventana de entrenamiento")
    minimo = ventana_entreno if minimo_observaciones is None else minimo_observaciones
    valores = matriz_retornos.to_numpy(dtype=float)
    columnas = matriz_retornos.columns
    n_filas = len(valores)

    trozos: list[pd.Series] = []
    fechas_elegidas: list = []
    registros: list[dict] = []
    for t in range(ventana_entreno, n_filas, paso):
        entreno = valores[t - ventana_entreno : t]
        fin = min(t + paso, n_filas)
        validas = np.sum(~np.isnan(entreno), axis=0) >= minimo
        indice_tramo = matriz_retornos.index[t:fin]
        fechas_elegidas.append(matriz_retornos.index[t])
        if not validas.any():
            trozos.append(pd.Series(np.nan, index=indice_tramo))
            registros.append({"columna": np.nan, "metrica_entreno": np.nan})
            continue
        puntuacion = np.asarray(metrica(entreno), dtype=float).copy()
        puntuacion[~validas | np.isnan(puntuacion)] = -np.inf
        elegida = int(np.argmax(puntuacion))
        trozos.append(pd.Series(valores[t:fin, elegida], index=indice_tramo))
        registros.append({"columna": columnas[elegida], "metrica_entreno": float(puntuacion[elegida])})

    serie = pd.concat(trozos)
    serie.name = "retorno_walk_forward"
    elecciones = pd.DataFrame(registros, index=pd.Index(fechas_elegidas, name=matriz_retornos.index.name))
    oos = serie.dropna().to_numpy()
    valor_metrica = float(np.asarray(metrica(oos[:, None]), dtype=float)[0]) if len(oos) else float("nan")
    return serie, elecciones, valor_metrica
