"""
Base classes para el módulo de señales.
========================================

SignalFrame — formato central de señales:
    pd.DataFrame
      Index   : DatetimeIndex (fechas de trading)
      Columns : ticker symbols ("AAPL", "MSFT", …)
      Values  : float [-1.0, +1.0]
                 -1 = short/underweight | 0 = neutral | +1 = long/overweight

VectorBT, backtest_engine y walk_forward consumen este formato directamente.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd

# Alias de tipo documentado
SignalFrame = pd.DataFrame


class BaseSignalGenerator(ABC):
    """
    Clase base abstracta para todos los generadores de señales.

    Contrato:
    - generate() recibe precios ajustados (DatetimeIndex × tickers) y devuelve
      un SignalFrame con las mismas fechas y tickers, valores ∈ [-1, +1].
    - Señales ausentes (NaN) son tratadas como neutral (0) por SignalCombiner.
    """

    @abstractmethod
    def generate(
        self,
        prices: pd.DataFrame,
        tickers: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Genera señales a partir de precios diarios ajustados.

        Args:
            prices: DataFrame con DatetimeIndex y columnas = tickers.
                    Valores = adjusted_close diario.
            tickers: Subconjunto opcional de tickers a procesar.
                     Si None, procesa todas las columnas de prices.
            **kwargs: Parámetros adicionales específicos del generador.

        Returns:
            SignalFrame: DataFrame con mismo índice que prices, columnas = tickers
                         procesados, valores ∈ [-1.0, +1.0].
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre corto legible del generador."""
        ...

    # ------------------------------------------------------------------
    # Helpers comunes
    # ------------------------------------------------------------------

    def _select_tickers(
        self,
        prices: pd.DataFrame,
        tickers: Optional[List[str]],
    ) -> List[str]:
        """Resuelve lista de tickers a procesar."""
        if tickers is None:
            return list(prices.columns)
        return [t for t in tickers if t in prices.columns]

    @staticmethod
    def _cross_sectional_rank_signal(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalización cross-seccional por rank → [-1, +1] con media exactamente 0.

        Para cada fecha (fila), rankea los valores entre tickers y mapea
        linealmente: rank 1 → -1, rank N → +1.  La media cross-seccional es 0
        para cualquier N ≥ 2 (la fórmula 2*(k-1)/(N-1)-1 produce una distribución
        simétrica alrededor de 0).

        Ventaja vs z-score: robusto a outliers y no requiere asumir distribución.
        """
        def _rank_row(row: pd.Series) -> pd.Series:
            valid = row.dropna()
            n = len(valid)
            if n <= 1:
                return row.where(row.notna(), 0.0) * 0.0
            ranked = valid.rank(method='average')  # 1 … N (ties → average rank)
            # Mapeo lineal: rank 1 → -1, rank N → +1  (media = 0 para cualquier N)
            signal = 2 * (ranked - 1) / (n - 1) - 1
            return signal.reindex(row.index)

        return df.apply(_rank_row, axis=1)

    @staticmethod
    def _zscore_clip_signal(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
        """
        Cross-sectional z-score → clipped a [-clip, +clip] → escalado a [-1, +1].

        Útil cuando la distribución de señales es aproximadamente normal.
        """
        mean = df.mean(axis=1)
        std  = df.std(axis=1).replace(0, np.nan)
        z = df.sub(mean, axis=0).div(std, axis=0)
        return (z.clip(-clip, clip) / clip)
