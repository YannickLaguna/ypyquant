"""
Asset Selection — Algoritmos puros de screening y diversificacion de activos.
==============================================================================

Responsabilidades:
- Screening de activos por Sharpe, momentum y volatilidad (acepta DataFrames pre-cargados)
- Diversificacion de universos por correlacion (4 algoritmos)
- Calculo de metricas de screening individuales por ticker

Interfaz:
- Todas las funciones reciben DataFrames de precios o retornos pre-calculados.
- Sin acceso a base de datos: la carga de datos queda en la capa de aplicacion (app/).

Se relaciona con:
- ypyquant.optimization.data_preprocessing: ProcesadorDatos, DiasTrading, anualizar_*
- app.services.portafolio.asset_selection: Wrapper con acceso a BD que llama estas funciones

Algoritmos de diversificacion:
  1. Greedy:      Seleccion secuencial evitando alta correlacion
  2. Clustering:  Clustering jerarquico + representante por cluster
  3. Distance:    Maximizacion de distancia promedio en espacio de correlaciones
  4. Mixed:       Combina scores externos con diversificacion greedy
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from ypyquant.optimization.data_preprocessing import (
    DiasTrading,
)

logger = logging.getLogger(__name__)

# ============================================================================
# METRICAS INDIVIDUALES DE SCREENING
# ============================================================================


def calcular_metricas_ticker(
    precios: pd.Series,
    tipo_retorno: Literal["log", "simple"] = "log",
    risk_free_rate: float = 0.02,
) -> dict:
    """
    Calcula metricas de screening para un ticker a partir de su serie de precios.

    Args:
        precios: Serie de precios (idealmente adjusted_close).
        tipo_retorno: 'log' para log-retornos, 'simple' para retornos simples.
        risk_free_rate: Tasa libre de riesgo anualizada para el Sharpe.

    Returns:
        Dict con: retorno_anual, volatilidad_anual, sharpe, sortino, momentum_score.
    """
    precios = precios.dropna()
    if len(precios) < 2:
        return {}

    if tipo_retorno == "log":
        retornos = np.log(precios / precios.shift(1)).dropna()
        retorno_anual = retornos.mean() * DiasTrading.ANUAL
    else:
        retornos = precios.pct_change().dropna()
        retorno_anual = (1 + retornos.mean()) ** DiasTrading.ANUAL - 1

    volatilidad_anual = retornos.std() * np.sqrt(DiasTrading.ANUAL)
    sharpe = (retorno_anual - risk_free_rate) / volatilidad_anual if volatilidad_anual > 0 else 0.0

    retornos_negativos = retornos[retornos < 0]
    downside_dev = retornos_negativos.std() * np.sqrt(DiasTrading.ANUAL)
    sortino = (retorno_anual - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0

    retorno_total = float(precios.iloc[-1] / precios.iloc[0]) - 1
    momentum_score = retorno_total / volatilidad_anual if volatilidad_anual > 0 else 0.0

    return {
        "retorno_anual": float(retorno_anual),
        "volatilidad_anual": float(volatilidad_anual),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "retorno_total": float(retorno_total),
        "momentum_score": float(momentum_score),
    }


# ============================================================================
# SCREENING (opera sobre matriz de precios: columnas = tickers, index = fechas)
# ============================================================================


def screen_by_sharpe(
    precios_df: pd.DataFrame,
    top_n: int = 10,
    tipo_retorno: Literal["log", "simple"] = "log",
    risk_free_rate: float = 0.02,
) -> list[dict]:
    """
    Ordena tickers por Sharpe Ratio historico y devuelve los top_n.

    Args:
        precios_df: DataFrame con precios (columnas = tickers).
        top_n: Numero de activos a devolver.
        tipo_retorno: Tipo de retornos para el calculo.
        risk_free_rate: Tasa libre de riesgo anualizada.

    Returns:
        Lista de dicts ordenada por sharpe desc: [{ticker, sharpe, retorno_anual, volatilidad_anual}]
    """
    resultados = []
    for ticker in precios_df.columns:
        metricas = calcular_metricas_ticker(
            precios_df[ticker], tipo_retorno=tipo_retorno, risk_free_rate=risk_free_rate
        )
        if metricas:
            resultados.append({"ticker": ticker, **metricas})

    resultados.sort(key=lambda x: x["sharpe"], reverse=True)
    return resultados[:top_n]


def screen_by_momentum(
    precios_df: pd.DataFrame,
    top_n: int = 10,
    tipo_retorno: Literal["log", "simple"] = "log",
) -> list[dict]:
    """
    Ordena tickers por momentum score (retorno_total / volatilidad) y devuelve top_n.

    Args:
        precios_df: DataFrame con precios (columnas = tickers).
        top_n: Numero de activos a devolver.
        tipo_retorno: Tipo de retornos para el calculo.

    Returns:
        Lista de dicts ordenada por momentum_score desc.
    """
    resultados = []
    for ticker in precios_df.columns:
        metricas = calcular_metricas_ticker(precios_df[ticker], tipo_retorno=tipo_retorno)
        if metricas:
            resultados.append({"ticker": ticker, **metricas})

    resultados.sort(key=lambda x: x["momentum_score"], reverse=True)
    return resultados[:top_n]


def screen_by_volatility(
    precios_df: pd.DataFrame,
    top_n: int = 10,
    tipo_retorno: Literal["log", "simple"] = "log",
    prefer_low: bool = True,
) -> list[dict]:
    """
    Ordena tickers por volatilidad anualizada.

    Args:
        precios_df: DataFrame con precios (columnas = tickers).
        top_n: Numero de activos a devolver.
        tipo_retorno: Tipo de retornos para el calculo.
        prefer_low: Si True, devuelve los de menor volatilidad; si False, los de mayor.

    Returns:
        Lista de dicts ordenada por volatilidad.
    """
    resultados = []
    for ticker in precios_df.columns:
        metricas = calcular_metricas_ticker(precios_df[ticker], tipo_retorno=tipo_retorno)
        if metricas:
            resultados.append({"ticker": ticker, **metricas})

    resultados.sort(key=lambda x: x["volatilidad_anual"], reverse=not prefer_low)
    return resultados[:top_n]


# ============================================================================
# DIVERSIFICACION POR CORRELACION (opera sobre matriz de retornos)
# ============================================================================


def diversify_greedy(
    retornos_df: pd.DataFrame,
    max_correlacion: float = 0.7,
    max_activos: int | None = None,
) -> list[str]:
    """
    Algoritmo Greedy: selecciona activos secuencialmente evitando alta correlacion.

    Empieza por el primer ticker y agrega candidatos que tengan correlacion
    menor a max_correlacion con todos los ya seleccionados.

    Args:
        retornos_df: DataFrame de retornos (columnas = tickers).
        max_correlacion: Correlacion maxima permitida entre activos seleccionados.
        max_activos: Limite de activos a seleccionar (None = sin limite).

    Returns:
        Lista de tickers seleccionados.
    """
    if retornos_df.empty or len(retornos_df.columns) < 2:
        return list(retornos_df.columns)

    corr = retornos_df.corr()
    seleccionados = [retornos_df.columns[0]]

    for ticker in retornos_df.columns[1:]:
        correlaciones = [abs(corr.loc[ticker, s]) for s in seleccionados]
        if max(correlaciones) < max_correlacion:
            seleccionados.append(ticker)
            if max_activos and len(seleccionados) >= max_activos:
                break

    return seleccionados


def diversify_clustering(
    retornos_df: pd.DataFrame,
    n_clusters: int = 5,
    method: str = 'average',
) -> list[str]:
    """
    Clustering Jerarquico: agrupa tickers por correlacion y devuelve un
    representante por cluster (el de menor correlacion media con su cluster).

    Args:
        retornos_df: DataFrame de retornos (columnas = tickers).
        n_clusters: Numero de clusters a formar.
        method: Metodo de linkage ('average', 'complete', 'single').

    Returns:
        Lista de tickers representativos (uno por cluster).
    """
    if retornos_df.empty or len(retornos_df.columns) < n_clusters:
        return list(retornos_df.columns)

    corr = retornos_df.corr()
    distancias = 1 - corr.abs()

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage=method,
    )
    etiquetas = clustering.fit_predict(distancias)

    seleccionados = []
    for cluster_id in range(n_clusters):
        tickers_cluster = retornos_df.columns[etiquetas == cluster_id].tolist()
        if len(tickers_cluster) == 1:
            seleccionados.append(tickers_cluster[0])
        else:
            # Representante: el de menor correlacion media con el resto del cluster
            corr_media = {
                t: corr.loc[t, [x for x in tickers_cluster if x != t]].abs().mean()
                for t in tickers_cluster
            }
            seleccionados.append(min(corr_media, key=corr_media.get))

    return seleccionados


def diversify_distance(
    retornos_df: pd.DataFrame,
    n_activos: int = 10,
    iteraciones: int = 100,
    semilla: int = 42,
) -> list[str]:
    """
    Maximizacion de Distancia: selecciona el subconjunto de tickers que maximiza
    la distancia promedio en el espacio de correlaciones mediante muestreo aleatorio.

    Args:
        retornos_df: DataFrame de retornos (columnas = tickers).
        n_activos: Numero de activos a seleccionar.
        iteraciones: Numero de muestras aleatorias a evaluar.
        semilla: Semilla para reproducibilidad.

    Returns:
        Lista de tickers que maximizan la distancia promedio mutua.
    """
    if retornos_df.empty or len(retornos_df.columns) <= n_activos:
        return list(retornos_df.columns)

    corr = retornos_df.corr()
    distancias = 1 - corr.abs()
    tickers = retornos_df.columns.tolist()
    rng = np.random.default_rng(semilla)

    mejor_seleccion: list[str] = []
    mejor_distancia = -1.0

    for _ in range(iteraciones):
        candidatos = rng.choice(tickers, size=min(n_activos, len(tickers)), replace=False).tolist()
        pares = [(candidatos[i], candidatos[j]) for i in range(len(candidatos)) for j in range(i + 1, len(candidatos))]
        if not pares:
            continue
        dist_media = float(np.mean([distancias.loc[a, b] for a, b in pares]))
        if dist_media > mejor_distancia:
            mejor_distancia = dist_media
            mejor_seleccion = candidatos

    return mejor_seleccion


def diversify_mixed(
    retornos_df: pd.DataFrame,
    scores: dict[str, float],
    max_correlacion: float = 0.7,
    top_n: int = 10,
) -> list[str]:
    """
    Score + Diversificacion: combina scores externos (Sharpe, factor scores, etc.)
    con el algoritmo greedy de correlacion.

    Primero ordena los tickers por score, luego aplica greedy sobre los top candidatos.

    Args:
        retornos_df: DataFrame de retornos (columnas = tickers).
        scores: Dict {ticker: score} (mayor score = preferido).
        max_correlacion: Correlacion maxima entre seleccionados.
        top_n: Numero maximo de activos a devolver.

    Returns:
        Lista de tickers seleccionados.
    """
    tickers_ordenados = sorted(
        [t for t in retornos_df.columns if t in scores],
        key=lambda t: scores[t],
        reverse=True,
    )
    # Considerar hasta 3x los candidatos para dejar margen al filtro de correlacion
    pool = tickers_ordenados[: top_n * 3]
    retornos_pool = retornos_df[pool].dropna(axis=1)

    return diversify_greedy(
        retornos_pool,
        max_correlacion=max_correlacion,
        max_activos=top_n,
    )


def calcular_matriz_correlacion(retornos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la matriz de correlacion entre activos.

    Args:
        retornos_df: DataFrame de retornos (columnas = tickers).

    Returns:
        DataFrame con la matriz de correlacion.
    """
    if retornos_df.empty:
        return pd.DataFrame()
    return retornos_df.corr()
