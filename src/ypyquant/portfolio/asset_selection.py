"""
Asset Selection — Screening y diversificación de activos (única fuente desde v0.3.0).
=====================================================================================

Responsabilidades:
- Funciones puras sobre DataFrames ya cargados (API de ypyquant 0.2.0, claves en español):
  calcular_metricas_ticker, screen_by_*, diversify_*, calcular_matriz_correlacion.
- `AssetSelectionService`: el servicio completo que tenía GrowSave
  (app/services/portafolio/asset_selection.py), con las mismas firmas y claves de resultado
  en inglés, pero sin BD: recibe un `cargador_precios` (ticker, inicio, fin) -> OHLCV y,
  opcionalmente, proveedores de universo histórico y de fundamentales.

Se relaciona con:
- ypyquant.data.matriz_precios: construir_matriz_precios / calcular_matriz_retornos.
- ypyquant.data.universos: `resolver_seleccion` y `universo_de_indice` (la resolución del universo, antes un
  `_resolver_universo` propio, vive allí).
- ypyquant.optimization.data_preprocessing: DiasTrading, TipoRetorno.
- GrowSave: su AssetSelectionService pasa a ser una subclase que inyecta get_ticker_data,
  TickerService.get_index_constituents_in_period y get_latest_fundamentals
  (ver docs/migracion-growsave-v0.3.0.md).

Algoritmos de diversificacion:
  1. Greedy:      Seleccion secuencial evitando alta correlacion
  2. Clustering:  Clustering jerarquico + representante por cluster
  3. Distance:    Maximizacion de distancia promedio en espacio de correlaciones
  4. Mixed:       Combina scores externos con diversificacion greedy
"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from ypyquant.data.matriz_precios import (
    CargadorPrecios,
    calcular_matriz_retornos,
    construir_matriz_precios,
    extraer_serie_precios,
)
from ypyquant.data.universos import resolver_seleccion, universo_de_indice
from ypyquant.optimization.data_preprocessing import (
    DiasTrading,
    TipoRetorno,
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
    # RandomState(semilla) reproduce exactamente np.random.seed(42) + np.random.choice de
    # GrowSave (resultados validados en producción) sin tocar el estado global de NumPy.
    rng = np.random.RandomState(semilla)

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


# ============================================================================
# SERVICIO COMPLETO (API de GrowSave, sin BD: datos inyectados)
# ============================================================================

# (index_name, fecha_inicio, fecha_fin, include_survivors_only) -> lista de tickers
ProveedorUniverso = Callable[[str, datetime, datetime, bool], list[str]]
# ticker -> dict de fundamentales (market_cap, pe_ratio, ...) o None
ProveedorFundamentales = Callable[[str], dict | None]

_PREFIJOS_FUNDAMENTALES = ('market_cap', 'pe_ratio', 'dividend', 'profit', 'return_on', 'debt_to')


class AssetSelectionService:
    """
    Servicio de selección de activos: screening (momentum, Sharpe, volatilidad, filtros
    técnicos y fundamentales) y 4 métodos de diversificación por correlación.

    Es el servicio de GrowSave sin acceso a BD. Los datos llegan por inyección:
        cargador_precios(ticker, inicio, fin) -> DataFrame OHLCV (columnas en minúscula o no)
        proveedor_universo(index_name, inicio_dt, fin_dt, include_survivors_only) -> [tickers]
        proveedor_fundamentales(ticker) -> dict | None
        proveedor_tickers_disponibles() -> [tickers]

    Usa log-retornos por defecto, como el resto del sistema de optimización. Los resultados
    conservan las claves en inglés de GrowSave ('ticker', 'sharpe_ratio', ...).
    """

    def __init__(
        self,
        cargador_precios: CargadorPrecios | None = None,
        tipo_retorno: Literal["log", "simple"] = "log",
        proveedor_universo: ProveedorUniverso | None = None,
        proveedor_fundamentales: ProveedorFundamentales | None = None,
        proveedor_tickers_disponibles: Callable[[], list[str]] | None = None,
    ):
        self.cargador_precios = cargador_precios
        self.tipo_retorno = tipo_retorno
        self._tipo_retorno_enum = TipoRetorno(tipo_retorno)
        self.proveedor_universo = proveedor_universo
        self.proveedor_fundamentales = proveedor_fundamentales
        self.proveedor_tickers_disponibles = proveedor_tickers_disponibles

    # ------------------------------------------------------------------
    # Acceso a datos (inyectado)
    # ------------------------------------------------------------------

    def _cargar_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """Carga el OHLCV de un ticker con el cargador inyectado."""
        if self.cargador_precios is None:
            raise RuntimeError("AssetSelectionService necesita un cargador_precios para cargar datos")
        return self.cargador_precios(ticker, start_date, end_date)

    def get_historical_universe(
        self,
        index_name: str,
        start_date: str,
        end_date: str,
        avoid_survivorship_bias: bool = True,
    ) -> list[str]:
        """
        Tickers que pertenecieron al índice en el período (evita survivorship bias).

        Requiere `proveedor_universo`; con avoid_survivorship_bias=True se incluyen los
        tickers que salieron del índice.
        """
        return universo_de_indice(index_name, start_date, end_date, avoid_survivorship_bias, self.proveedor_universo)

    def get_available_tickers_info(self) -> list[dict]:
        """Lista de tickers disponibles con información básica (nombre/sector aún sin datos)."""
        tickers = self.proveedor_tickers_disponibles() if self.proveedor_tickers_disponibles else []
        return [{'symbol': t, 'name': t, 'sector': None, 'market_cap': None} for t in tickers]

    def _get_prices_matrix(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Matriz de precios ajustados (vacía si no hay datos)."""
        if self.cargador_precios is None:
            raise RuntimeError("AssetSelectionService necesita un cargador_precios para cargar datos")
        return construir_matriz_precios(tickers, start_date, end_date, self._cargar_ohlcv, estricto=False)

    def _get_returns_matrix(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        tipo_retorno: Literal["log", "simple"] | None = None,
    ) -> pd.DataFrame:
        """Matriz de retornos diarios limpia (ProcesadorDatos, 90 % de datos válidos mínimo)."""
        tipo = tipo_retorno or self.tipo_retorno
        precios = self._get_prices_matrix(tickers, start_date, end_date)
        return calcular_matriz_retornos(precios, tipo_retorno=tipo, ratio_minimo_validos=0.90, estricto=False)

    def _calcular_retornos_ticker(self, prices: pd.Series) -> pd.Series:
        """Retornos de un ticker con el tipo configurado (log o simple)."""
        if self.tipo_retorno == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        return returns.dropna()

    def _anualizar_media(self, media_retornos: float) -> float:
        """Log: media * 252 (aditiva). Simple: (1 + media)^252 - 1 (compuesta)."""
        if self.tipo_retorno == "log":
            return media_retornos * DiasTrading.ANUAL
        return (1 + media_retornos) ** DiasTrading.ANUAL - 1

    def _anualizar_volatilidad(self, volatilidad: float) -> float:
        """Volatilidad diaria * sqrt(252)."""
        return volatilidad * np.sqrt(DiasTrading.ANUAL)

    def _precios_ticker(self, ticker: str, start_date: str, end_date: str, min_filas: int = 1) -> pd.Series | None:
        """Serie de precios ajustados de un ticker, o None si no hay al menos `min_filas`."""
        ohlcv = self._cargar_ohlcv(ticker, start_date, end_date)
        if ohlcv is None or ohlcv.empty or len(ohlcv) < min_filas:
            return None
        return extraer_serie_precios(ohlcv, ticker, avisar=False)

    # ------------------------------------------------------------------
    # Screening básico
    # ------------------------------------------------------------------

    def screen_by_momentum(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        top_n: int = 10,
        lookback_days: int = 252,
        index_name: str | None = None,
        avoid_survivorship_bias: bool = True,
    ) -> list[dict]:
        """
        Activos con mejor momentum (retorno total / volatilidad anualizada).

        Returns:
            [{ticker, total_return, annualized_volatility, sharpe_ratio, momentum_score}]
            ordenada por momentum_score descendente (top_n). sharpe_ratio aquí es sin r_f.
        """
        tickers = resolver_seleccion(tickers, start_date, end_date, index_name, avoid_survivorship_bias, self.proveedor_universo)
        datos = []
        for ticker in tickers:
            try:
                prices = self._precios_ticker(ticker, start_date, end_date, min_filas=lookback_days)
                if prices is None:
                    continue
                total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                returns = self._calcular_retornos_ticker(prices)
                volatility = self._anualizar_volatilidad(returns.std())
                mean_return = self._anualizar_media(returns.mean())
                datos.append({
                    'ticker': ticker,
                    'total_return': float(total_return),
                    'annualized_volatility': float(volatility),
                    'sharpe_ratio': float(mean_return / volatility) if volatility > 0 else 0,
                    'momentum_score': float(total_return / volatility) if volatility > 0 else 0,
                })
            except Exception as exc:  # noqa: BLE001 — un ticker defectuoso no debe abortar el lote
                logger.debug(f"Error calculando momentum de {ticker}: {exc}")
        datos.sort(key=lambda x: x['momentum_score'], reverse=True)
        return datos[:top_n]

    def screen_by_sharpe(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        top_n: int = 10,
        risk_free_rate: float | None = None,
        index_name: str | None = None,
        avoid_survivorship_bias: bool = True,
    ) -> list[dict]:
        """
        Activos por Sharpe histórico ((media anual − r_f) / volatilidad anual).

        Returns:
            [{ticker, sharpe_ratio, annualized_return, annualized_volatility}] (top_n).
        """
        tickers = resolver_seleccion(tickers, start_date, end_date, index_name, avoid_survivorship_bias, self.proveedor_universo)
        if risk_free_rate is None:
            risk_free_rate = 0.02  # Valor conservador por defecto
        datos = []
        for ticker in tickers:
            try:
                prices = self._precios_ticker(ticker, start_date, end_date)
                if prices is None:
                    continue
                returns = self._calcular_retornos_ticker(prices)
                mean_return = self._anualizar_media(returns.mean())
                volatility = self._anualizar_volatilidad(returns.std())
                datos.append({
                    'ticker': ticker,
                    'sharpe_ratio': float((mean_return - risk_free_rate) / volatility) if volatility > 0 else 0.0,
                    'annualized_return': float(mean_return),
                    'annualized_volatility': float(volatility),
                })
            except Exception as exc:  # noqa: BLE001 — un ticker defectuoso no debe abortar el lote
                logger.debug(f"Error calculando Sharpe de {ticker}: {exc}")
        datos.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        return datos[:top_n]

    def screen_by_volatility(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        top_n: int = 10,
        prefer_low: bool = True,
        index_name: str | None = None,
        avoid_survivorship_bias: bool = True,
    ) -> list[dict]:
        """
        Activos por volatilidad anualizada (baja si prefer_low, alta si no).

        Returns:
            [{ticker, annualized_volatility, annualized_return}] (top_n).
        """
        tickers = resolver_seleccion(tickers, start_date, end_date, index_name, avoid_survivorship_bias, self.proveedor_universo)
        datos = []
        for ticker in tickers:
            try:
                prices = self._precios_ticker(ticker, start_date, end_date)
                if prices is None:
                    continue
                returns = self._calcular_retornos_ticker(prices)
                datos.append({
                    'ticker': ticker,
                    'annualized_volatility': float(self._anualizar_volatilidad(returns.std())),
                    'annualized_return': float(self._anualizar_media(returns.mean())),
                })
            except Exception as exc:  # noqa: BLE001 — un ticker defectuoso no debe abortar el lote
                logger.debug(f"Error calculando volatilidad de {ticker}: {exc}")
        datos.sort(key=lambda x: x['annualized_volatility'], reverse=not prefer_low)
        return datos[:top_n]

    # ------------------------------------------------------------------
    # Diversificación por correlación (cargan retornos y delegan en las funciones puras)
    # ------------------------------------------------------------------

    def diversify_greedy(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        max_correlation: float = 0.7,
        max_assets: int | None = None,
    ) -> list[str]:
        """Método 1 (greedy): añade activos con |correlación| < max_correlation con los elegidos."""
        retornos = self._get_returns_matrix(tickers, start_date, end_date)
        return diversify_greedy(retornos, max_correlacion=max_correlation, max_activos=max_assets)

    def diversify_clustering(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        n_clusters: int = 5,
        method: str = 'average',
    ) -> list[str]:
        """Método 2 (clustering jerárquico): un representante por cluster."""
        retornos = self._get_returns_matrix(tickers, start_date, end_date)
        return diversify_clustering(retornos, n_clusters=n_clusters, method=method)

    def diversify_distance(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        n_assets: int = 10,
        iterations: int = 100,
    ) -> list[str]:
        """Método 3 (máxima distancia media 1 − |ρ|) por muestreo aleatorio con semilla 42."""
        retornos = self._get_returns_matrix(tickers, start_date, end_date)
        return diversify_distance(retornos, n_activos=n_assets, iteraciones=iterations, semilla=42)

    def diversify_mixed(
        self,
        tickers: list[str],
        scores: dict[str, float],
        start_date: str,
        end_date: str,
        max_correlation: float = 0.7,
        top_n: int = 10,
    ) -> list[str]:
        """Método 4 (score + greedy): greedy sobre los 3·top_n tickers de mayor score."""
        ordenados = sorted([t for t in tickers if t in scores], key=lambda t: scores[t], reverse=True)
        return self.diversify_greedy(
            tickers=ordenados[:top_n * 3],
            start_date=start_date,
            end_date=end_date,
            max_correlation=max_correlation,
            max_assets=top_n,
        )

    def calculate_correlation_matrix(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Matriz de correlación de los retornos (vacía si no hay datos)."""
        return calcular_matriz_correlacion(self._get_returns_matrix(tickers, start_date, end_date))

    # ------------------------------------------------------------------
    # Screening personalizado y por filtros
    # ------------------------------------------------------------------

    def screen_custom(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        criteria: dict[str, Any],
    ) -> list[dict]:
        """
        Screening con criterios min_return, max_volatility, min_sharpe y max_correlation
        (este último aplica diversify_greedy al final). Retornos simples, r_f = 2 %.

        Returns:
            [{ticker, annualized_return, annualized_volatility, sharpe_ratio}] que cumplen.
        """
        resultados = []
        for ticker in tickers:
            try:
                prices = self._precios_ticker(ticker, start_date, end_date)
                if prices is None:
                    continue
                returns = prices.pct_change().dropna()
                mean_return = returns.mean() * 252
                volatility = returns.std() * np.sqrt(252)
                sharpe = (mean_return - 0.02) / volatility if volatility > 0 else 0

                cumple = True
                if 'min_return' in criteria and mean_return < criteria['min_return']:
                    cumple = False
                if 'max_volatility' in criteria and volatility > criteria['max_volatility']:
                    cumple = False
                if 'min_sharpe' in criteria and sharpe < criteria['min_sharpe']:
                    cumple = False
                if cumple:
                    resultados.append({
                        'ticker': ticker,
                        'annualized_return': float(mean_return),
                        'annualized_volatility': float(volatility),
                        'sharpe_ratio': float(sharpe),
                    })
            except Exception as exc:  # noqa: BLE001 — un ticker defectuoso no debe abortar el lote
                logger.warning(f"Error en screening de {ticker}: {exc}")

        if 'max_correlation' in criteria and resultados:
            diversificados = self.diversify_greedy(
                [r['ticker'] for r in resultados], start_date, end_date, criteria['max_correlation']
            )
            resultados = [r for r in resultados if r['ticker'] in diversificados]
        return resultados

    def screen_by_filters(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        filters: dict[str, Any],
        top_n: int | None = None,
        apply_correlation_filter: bool = False,
        max_correlation: float = 0.7,
        risk_free_rate: float = 0.02,
    ) -> list[dict]:
        """
        Screening por filtros técnicos (sharpe/sortino/calmar mín., retorno y volatilidad
        mín./máx., max_drawdown_max, beta_min/max) y fundamentales (market_cap, pe_ratio,
        dividend_yield, profit_margin, return_on_equity, debt_to_equity). Los fundamentales
        solo se piden si hay algún filtro fundamental y hay `proveedor_fundamentales`.

        Returns:
            Activos que pasan, ordenados por Sharpe desc., limitados a top_n y, si se pide,
            diversificados por correlación.
        """
        resultados = []
        logger.info(f"Screening {len(tickers)} tickers con {len(filters)} filtros activos...")
        pide_fundamentales = any(k.startswith(_PREFIJOS_FUNDAMENTALES) for k in filters)

        for ticker in tickers:
            try:
                prices = self._precios_ticker(ticker, start_date, end_date, min_filas=30)
                if prices is None:
                    continue
                returns = prices.pct_change().dropna()
                if len(returns) < 30:
                    continue

                mean_return = returns.mean() * 252
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0

                downside = returns[returns < 0]
                downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else volatility
                sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0

                acumulado = (1 + returns).cumprod()
                drawdown = (acumulado - acumulado.cummax()) / acumulado.cummax()
                max_drawdown = float(drawdown.min())
                calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
                beta = None  # Pendiente: beta frente al índice de referencia

                metricas = {
                    'ticker': ticker,
                    'annualized_return': float(mean_return),
                    'annualized_volatility': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'sortino_ratio': float(sortino_ratio),
                    'calmar_ratio': float(calmar_ratio),
                    'max_drawdown': float(max_drawdown),
                    'beta': beta,
                }

                fundamentales: dict[str, Any] = {}
                if pide_fundamentales and self.proveedor_fundamentales is not None:
                    datos_fund = self.proveedor_fundamentales(ticker)
                    if datos_fund:
                        fundamentales = {
                            clave: datos_fund.get(clave)
                            for clave in ('market_cap', 'pe_ratio', 'dividend_yield', 'profit_margin',
                                          'return_on_equity', 'return_on_assets', 'debt_to_equity',
                                          'price_to_book', 'peg_ratio')
                        }
                metricas.update(fundamentales)

                if _pasa_filtros(filters, metricas, fundamentales):
                    resultados.append(metricas)
            except Exception as exc:  # noqa: BLE001 — un ticker defectuoso no debe abortar el lote
                logger.warning(f"Error en screening de {ticker}: {exc}")

        resultados.sort(key=lambda x: x.get('sharpe_ratio', -999), reverse=True)
        if top_n and len(resultados) > top_n:
            resultados = resultados[:top_n]

        if apply_correlation_filter and len(resultados) > 1:
            diversificados = self.diversify_greedy(
                [r['ticker'] for r in resultados], start_date, end_date, max_correlation
            )
            resultados = [r for r in resultados if r['ticker'] in diversificados]

        logger.info(f"Screening completado: {len(resultados)} activos seleccionados")
        return resultados

    def get_screening_presets(self) -> dict[str, dict]:
        """Presets de screening predefinidos (los que ofrece la API de GrowSave)."""
        return {
            'momentum_growth': {
                'name': 'Momentum & Growth',
                'description': 'Activos con alto momentum y crecimiento',
                'method': 'momentum',
                'params': {'top_n': 10, 'lookback_days': 252},
            },
            'low_volatility': {
                'name': 'Low Volatility',
                'description': 'Activos con baja volatilidad',
                'method': 'volatility',
                'params': {'top_n': 10, 'prefer_low': True},
            },
            'high_sharpe': {
                'name': 'High Sharpe Ratio',
                'description': 'Mejor relación riesgo-retorno',
                'method': 'sharpe',
                'params': {'top_n': 10, 'risk_free_rate': 0.02},
            },
            'balanced': {
                'name': 'Balanced',
                'description': 'Balance entre retorno, riesgo y diversificación',
                'method': 'custom',
                'params': {'min_return': 0.05, 'max_volatility': 0.30, 'min_sharpe': 0.3, 'max_correlation': 0.7},
            },
        }


def _pasa_filtros(filtros: dict[str, Any], metricas: dict[str, Any], fundamentales: dict[str, Any]) -> bool:
    """
    Aplica los filtros de screen_by_filters. Los fundamentales sin dato (None) no descartan.

    Clave de filtro '<métrica>_min' exige métrica >= valor; '<métrica>_max', métrica <= valor.
    max_drawdown_max: el drawdown (negativo) debe ser >= el valor (p. ej. -0.25).
    """
    tecnicos = {
        'sharpe_ratio_min': ('sharpe_ratio', 'min'),
        'sortino_ratio_min': ('sortino_ratio', 'min'),
        'calmar_ratio_min': ('calmar_ratio', 'min'),
        'annualized_return_min': ('annualized_return', 'min'),
        'annualized_return_max': ('annualized_return', 'max'),
        'annualized_volatility_min': ('annualized_volatility', 'min'),
        'annualized_volatility_max': ('annualized_volatility', 'max'),
        'max_drawdown_max': ('max_drawdown', 'min'),
        'beta_min': ('beta', 'min'),
        'beta_max': ('beta', 'max'),
    }
    fundamentales_filtros = {
        'market_cap_min': ('market_cap', 'min'),
        'market_cap_max': ('market_cap', 'max'),
        'pe_ratio_min': ('pe_ratio', 'min'),
        'pe_ratio_max': ('pe_ratio', 'max'),
        'dividend_yield_min': ('dividend_yield', 'min'),
        'profit_margin_min': ('profit_margin', 'min'),
        'return_on_equity_min': ('return_on_equity', 'min'),
        'debt_to_equity_max': ('debt_to_equity', 'max'),
    }
    for nombre, (clave, sentido) in {**tecnicos, **fundamentales_filtros}.items():
        if nombre not in filtros:
            continue
        origen = fundamentales if nombre in fundamentales_filtros else metricas
        valor = origen.get(clave)
        if valor is None:
            continue
        if sentido == 'min' and valor < filtros[nombre]:
            return False
        if sentido == 'max' and valor > filtros[nombre]:
            return False
    return True
