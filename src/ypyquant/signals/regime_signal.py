"""
RegimeSignalGenerator — filtro de régimen de mercado.
======================================================

Wraps MarketRegimeClassifier para producir un multiplicador direccional:
    Bull  (+2) → señal = +1.0
    Lateral   → señal =  0.0
    Bear  (0) → señal = -1.0
    High Vol  → señal =  0.0  (cautela — no tomar posición direccional)

Este generador actúa como filtro, no como señal standalone.
SignalCombiner lo usa con peso bajo (ej. 0.1) para escalar otras señales.

La señal es uniforme a todos los tickers (régimen del mercado global).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ypyquant.signals.base import BaseSignalGenerator, SignalFrame

logger = logging.getLogger(__name__)

# Mapeo régimen → señal
_REGIME_TO_SIGNAL = {
    0: -1.0,   # Bear Market
    1:  0.0,   # Lateral/Consolidation
    2: +1.0,   # Bull Market
    3:  0.0,   # High Volatility → neutral
}


class RegimeSignalGenerator(BaseSignalGenerator):
    """
    Genera una señal de régimen uniform para todos los tickers.

    La señal es scalar por fecha (mismo valor para todos los tickers),
    derivada del régimen del mercado representativo (market_returns).

    Uso típico: combinar con weight bajo para escalar señales técnicas/factor
    según el entorno de mercado.
    """

    @property
    def name(self) -> str:
        return "regime"

    def generate(
        self,
        prices: pd.DataFrame,
        tickers: list[str] | None = None,
        market_returns: pd.Series | None = None,
        model_type: str = "random_forest",
        lookback: int = 60,
        **kwargs,
    ) -> SignalFrame:
        """
        Args:
            prices: adjusted_close diario (DatetimeIndex × tickers).
            tickers: Subconjunto de tickers. None → todos.
            market_returns: Retornos diarios del índice de mercado (ej. S&P 500).
                            Si None, se usa la media de retornos de prices como proxy.
            model_type: 'svm' o 'random_forest' para MarketRegimeClassifier.
            lookback: Ventana rolling para features del clasificador (días).

        Returns:
            SignalFrame: señal uniform ∈ {-1, 0, +1} para todos los tickers.
        """
        try:
            from ypyquant.ml.market_regime import MarketRegimeClassifier
        except ImportError as e:
            logger.warning(f"[RegimeSignal] Dependencia no disponible: {e}. Señal=0.")
            cols = self._select_tickers(prices, tickers)
            return pd.DataFrame(0.0, index=prices.index, columns=cols)

        cols = self._select_tickers(prices, tickers)

        # Proxy de mercado si no se provee
        if market_returns is None:
            mkt = prices[cols].pct_change().mean(axis=1).dropna()
        else:
            mkt = market_returns.dropna()

        if len(mkt) < lookback + 100:
            logger.warning(
                f"[RegimeSignal] Datos insuficientes ({len(mkt)} días < {lookback + 100}). Señal=0."
            )
            return pd.DataFrame(0.0, index=prices.index, columns=cols)

        try:
            clf = MarketRegimeClassifier(model_type=model_type)
            clf.train(mkt, lookback=lookback)

            # Predecir régimen para cada fecha usando ventana rolling
            regimen_features = clf.create_features(mkt, lookback=lookback)
            regimenes = clf.model.predict(regimen_features)
            regime_series = pd.Series(regimenes, index=regimen_features.index)

            # Mapear a señal escalar
            señal_series = regime_series.map(_REGIME_TO_SIGNAL).fillna(0.0)

            # Broadcast a todos los tickers
            signal_df = pd.DataFrame(
                np.nan, index=prices.index, columns=cols
            )
            for ticker in cols:
                signal_df[ticker] = señal_series.reindex(prices.index)

            return signal_df.fillna(0.0).clip(-1.0, 1.0)

        except Exception as exc:
            logger.warning(f"[RegimeSignal] Error en clasificación: {exc}. Señal=0.")
            return pd.DataFrame(0.0, index=prices.index, columns=cols)
