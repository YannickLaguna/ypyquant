"""
SignalCombiner — ensemble ponderado de generadores de señales.
==============================================================

Combina múltiples SignalFrames en uno solo mediante media ponderada.

Convenciones:
- NaN en señales individuales → tratado como 0.0 antes de combinar
  (señal neutral, no contamina el promedio).
- Pesos normalizados automáticamente a suma 1.
- Resultado recortado a [-1, +1].
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ypyquant.signals.base import BaseSignalGenerator, SignalFrame

logger = logging.getLogger(__name__)


class SignalCombiner:
    """
    Combina señales de múltiples BaseSignalGenerator en un único SignalFrame.

    Ejemplo de uso::

        combiner = SignalCombiner(
            generators=[TechnicalSignalGenerator(), FactorSignalGenerator()],
            weights=[0.6, 0.4],
        )
        signal_frame = combiner.generate(prices_df, factor_data=ff_data)

    El SignalFrame resultante puede alimentar directamente a:
    - VectorBT (scripts/vbt_signal_scan.py)
    - backtest_engine.py
    - walk_forward.py (como señales de entrada a la fase de generación)
    """

    SUPPORTED_METHODS = ("weighted_mean", "median", "majority_vote")

    def __init__(
        self,
        generators: List[BaseSignalGenerator],
        weights: Optional[List[float]] = None,
        combination_method: str = "weighted_mean",
        fillna_value: float = 0.0,
    ):
        """
        Args:
            generators: Lista de generadores de señales.
            weights: Pesos por generador. None → equal-weight.
                     Se normalizan a suma 1 automáticamente.
            combination_method: Método de combinación:
                - "weighted_mean": media ponderada (default).
                - "median": mediana (más robusta a outliers).
                - "majority_vote": signo de la mayoría ponderada.
            fillna_value: Valor para NaN antes de combinar (default: 0.0 = neutral).
        """
        if not generators:
            raise ValueError("Se requiere al menos un generador de señales.")

        if combination_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"combination_method '{combination_method}' no soportado. "
                f"Opciones: {self.SUPPORTED_METHODS}"
            )

        self.generators = generators
        self.combination_method = combination_method
        self.fillna_value = fillna_value

        # Normalizar pesos
        if weights is None:
            n = len(generators)
            self.weights = [1.0 / n] * n
        else:
            if len(weights) != len(generators):
                raise ValueError(
                    f"len(weights)={len(weights)} != len(generators)={len(generators)}"
                )
            total = sum(weights)
            if total <= 0:
                raise ValueError("La suma de pesos debe ser positiva.")
            self.weights = [w / total for w in weights]

    @property
    def name(self) -> str:
        names = "+".join(g.name for g in self.generators)
        return f"combiner({names})"

    def generate(
        self,
        prices: pd.DataFrame,
        tickers: Optional[List[str]] = None,
        **kwargs,
    ) -> SignalFrame:
        """
        Genera el SignalFrame combinado.

        Args:
            prices: adjusted_close diario (DatetimeIndex × tickers).
            tickers: Subconjunto de tickers. None → todos los de prices.
            **kwargs: Se pasan a todos los generadores (ej. factor_data=...).

        Returns:
            SignalFrame combinado ∈ [-1, +1].
        """
        cols = tickers if tickers else list(prices.columns)
        señales_individuales: List[pd.DataFrame] = []

        for i, gen in enumerate(self.generators):
            try:
                sf = gen.generate(prices, tickers=cols, **kwargs)
                # Asegurar que tiene el índice y columnas correctos
                sf = sf.reindex(index=prices.index, columns=cols).fillna(self.fillna_value)
                señales_individuales.append(sf)
                logger.debug(f"[Combiner] {gen.name}: shape={sf.shape}, mean={sf.mean().mean():.4f}")
            except Exception as exc:
                logger.warning(
                    f"[Combiner] Generador '{gen.name}' falló: {exc}. "
                    "Sustituyendo con señal cero."
                )
                señales_individuales.append(
                    pd.DataFrame(self.fillna_value, index=prices.index, columns=cols)
                )

        return self._combinar(señales_individuales, prices.index, cols)

    # ------------------------------------------------------------------
    # Métodos de combinación
    # ------------------------------------------------------------------

    def _combinar(
        self,
        señales: List[pd.DataFrame],
        index: pd.DatetimeIndex,
        columns: List[str],
    ) -> SignalFrame:
        if not señales:
            return pd.DataFrame(0.0, index=index, columns=columns)

        if self.combination_method == "weighted_mean":
            combined = pd.DataFrame(0.0, index=index, columns=columns)
            for sf, w in zip(señales, self.weights):
                combined = combined + sf * w

        elif self.combination_method == "median":
            # Mediana ponderada: apilar y calcular mediana
            stack = np.stack([sf.values for sf in señales], axis=0)
            median_vals = np.median(stack, axis=0)
            combined = pd.DataFrame(median_vals, index=index, columns=columns)

        elif self.combination_method == "majority_vote":
            # Voto ponderado: suma de signos ponderados
            combined = pd.DataFrame(0.0, index=index, columns=columns)
            for sf, w in zip(señales, self.weights):
                combined = combined + np.sign(sf) * w
            combined = np.sign(combined)  # -1, 0, +1

        else:
            raise ValueError(f"Método desconocido: {self.combination_method}")

        return combined.clip(-1.0, 1.0)

    def get_weights_summary(self) -> Dict[str, float]:
        """Devuelve dict {nombre_generador: peso} para logging/debugging."""
        return {g.name: w for g, w in zip(self.generators, self.weights)}
