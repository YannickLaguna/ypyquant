"""
TechnicalSignalGenerator — señales técnicas sobre precios ajustados.
=====================================================================

Señales implementadas (todas normalizadas a [-1, +1] vía rank cross-seccional):
  1. RSI(14) — z-score cross-seccional; alto RSI = sobrecompra → señal negativa (contrarian).
  2. MACD normalizado por precio — momentum cross-seccional.
  3. Bollinger %B — posición relativa en las bandas (mean-reversion).
  4. Momentum 12-1 — retorno 12m – retorno 1m (skip-month).

Combinación interna: media aritmética de las 4 señales, clip final a [-1, 1].
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ypyquant.signals.base import BaseSignalGenerator, SignalFrame

logger = logging.getLogger(__name__)


class TechnicalSignalGenerator(BaseSignalGenerator):
    """
    Genera señales técnicas para un universo multi-ticker.

    Input:  prices — DataFrame(DatetimeIndex × tickers, adjusted_close).
    Output: SignalFrame(DatetimeIndex × tickers, float ∈ [-1, +1]).
    """

    @property
    def name(self) -> str:
        return "technical"

    def generate(
        self,
        prices: pd.DataFrame,
        tickers: list[str] | None = None,
        rsi_window: int = 14,
        bb_window: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        **kwargs,
    ) -> SignalFrame:
        """
        Args:
            prices: adjusted_close diario (DatetimeIndex × tickers).
            tickers: Subconjunto de tickers. None → todos los de prices.
            rsi_window: Ventana del RSI (default 14).
            bb_window: Ventana Bollinger Bands (default 20).
            macd_fast/slow/signal: Parámetros del MACD.

        Returns:
            SignalFrame con valores ∈ [-1, 1].
        """
        cols = self._select_tickers(prices, tickers)
        px = prices[cols].copy()

        # Señales individuales (cada una ya en [-1,1] vía rank cross-seccional)
        s_rsi      = self._rsi_signal(px, rsi_window)
        s_macd     = self._macd_signal(px, macd_fast, macd_slow, macd_signal)
        s_bb       = self._bollinger_signal(px, bb_window)
        s_momentum = self._momentum_signal(px)

        # Media aritmética de las señales disponibles, clip final
        combined = pd.concat([s_rsi, s_macd, s_bb, s_momentum], axis=0, keys=["rsi", "macd", "bb", "mom"])
        signal = combined.groupby(level=1).mean()  # promedia sobre indicadores por fecha

        # Alternativa más clara: alinear y promediar
        signals_list = [s for s in [s_rsi, s_macd, s_bb, s_momentum] if s is not None and not s.empty]
        if not signals_list:
            return pd.DataFrame(0.0, index=px.index, columns=cols)

        aligned = pd.concat(signals_list, axis=0)
        # Reshape: cada señal tiene mismo shape (dates x tickers), stack y mean
        signals_stack = pd.concat(signals_list, keys=range(len(signals_list)))
        signal = signals_stack.groupby(level=1).mean()

        return signal.clip(-1.0, 1.0).reindex(index=px.index, columns=cols)

    # ------------------------------------------------------------------
    # Señales individuales
    # ------------------------------------------------------------------

    def _rsi_signal(self, prices: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        RSI(14) invertido (contrarian): alto RSI → sobreventa de señal → señal negativa.
        Normalización: rank cross-seccional → [-1, +1], luego invertir signo.
        """
        rsi_df = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        for col in prices.columns:
            px = prices[col].dropna()
            if len(px) < window + 1:
                continue
            delta = px.diff()
            gain = delta.clip(lower=0).rolling(window).mean()
            loss = (-delta.clip(upper=0)).rolling(window).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            rsi_df[col] = rsi.reindex(prices.index)

        # RSI alto = sobrecompra → señal negativa (contrarian mean-reversion)
        ranked = self._cross_sectional_rank_signal(rsi_df.astype(float))
        return -ranked  # invertir: alto RSI relativo → señal negativa

    def _macd_signal(
        self,
        prices: pd.DataFrame,
        fast: int,
        slow: int,
        signal_window: int,
    ) -> pd.DataFrame:
        """MACD / precio → rank cross-seccional → momentum signal."""
        macd_norm = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        for col in prices.columns:
            px = prices[col].dropna()
            if len(px) < slow + signal_window:
                continue
            ema_fast = px.ewm(span=fast, adjust=False).mean()
            ema_slow = px.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            # Normalizar por precio para comparar cross-seccional
            macd_norm[col] = (macd_line / px.replace(0, np.nan)).reindex(prices.index)

        return self._cross_sectional_rank_signal(macd_norm.astype(float))

    def _bollinger_signal(self, prices: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Bollinger %B mean-reversion:
        - %B = (precio - lower) / (upper - lower)
        - signal = 1 - 2*%B  →  precio en lower → +1 (compra), precio en upper → -1 (venta).
        """
        bb_signal = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        for col in prices.columns:
            px = prices[col].dropna()
            if len(px) < window:
                continue
            ma    = px.rolling(window).mean()
            std   = px.rolling(window).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            band_width = (upper - lower).replace(0, np.nan)
            pct_b = (px - lower) / band_width
            bb_signal[col] = (1 - 2 * pct_b).reindex(prices.index)

        return bb_signal.astype(float).clip(-1.0, 1.0)

    def _momentum_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum 12-1: retorno de 12 meses menos retorno de 1 mes (skip-month).
        Cross-sectional rank → [-1, +1].
        Aprox: 252 días (12m) - 21 días (1m) lookback.
        """
        ret_12m = prices.pct_change(252)
        ret_1m  = prices.pct_change(21)
        mom     = ret_12m - ret_1m
        return self._cross_sectional_rank_signal(mom)
