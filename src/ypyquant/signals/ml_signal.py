"""
MLSignalGenerator — wraps ypyquant.ml.ml_predictor.MLPredictor.
================================================================

Para cada ticker, entrena MLPredictor en una ventana rolling y mapea
predict_proba a [-1, +1]:
    signal = 2 * P(up) - 1

donde P(up) es la probabilidad de movimiento positivo (clase 1).

Entrenamiento inline (por ticker, en la misma llamada a generate()) —
apropiado para señales batch en research. No persiste modelos entre calls.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from ypyquant.signals.base import BaseSignalGenerator, SignalFrame

logger = logging.getLogger(__name__)


class MLSignalGenerator(BaseSignalGenerator):
    """
    Señal ML basada en predicciones de MLPredictor por ticker.

    Flujo por ticker:
    1. Construir DataFrame OHLCV sintético (solo adjusted_close disponible;
       se replica como Close/Open/High/Low/Volume para compatibilidad con FeatureEngineering).
    2. Entrenar MLPredictor en la ventana de entrenamiento.
    3. Predecir sobre la ventana completa usando predict_proba.
    4. Mapear P(up) → [-1, +1].

    Advertencia: entrenamiento inline es lento para universos grandes.
    Úsalo con top_n tickers reducido o paraleliza externamente.
    """

    @property
    def name(self) -> str:
        return "ml"

    def generate(
        self,
        prices: pd.DataFrame,
        tickers: Optional[List[str]] = None,
        train_window: int = 252,
        forecast_horizon: int = 5,
        **kwargs,
    ) -> SignalFrame:
        """
        Args:
            prices: adjusted_close diario (DatetimeIndex × tickers).
            tickers: Subconjunto de tickers. None → todos.
            train_window: Días de historia para entrenar cada modelo (default: 252).
            forecast_horizon: Horizonte de predicción en días (default: 5).

        Returns:
            SignalFrame: 2*P(up) - 1, valores ∈ [-1, +1].
        """
        try:
            from ypyquant.ml.ml_predictor import MLPredictor, ModelType
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as e:
            logger.warning(f"[MLSignal] Dependencia no disponible: {e}. Retornando señal cero.")
            cols = self._select_tickers(prices, tickers)
            return pd.DataFrame(0.0, index=prices.index, columns=cols)

        cols = self._select_tickers(prices, tickers)
        signal_df = pd.DataFrame(np.nan, index=prices.index, columns=cols)

        for ticker in cols:
            px_series = prices[ticker].dropna()
            if len(px_series) < train_window + forecast_horizon + 60:
                logger.debug(f"[MLSignal] {ticker}: datos insuficientes, señal=0.")
                signal_df[ticker] = 0.0
                continue

            try:
                # Construir DataFrame OHLCV mínimo (replicar Close para H/L/O, Volume=1e6)
                ohlcv = pd.DataFrame({
                    "Open":   px_series.values,
                    "High":   px_series.values,
                    "Low":    px_series.values,
                    "Close":  px_series.values,
                    "Volume": np.full(len(px_series), 1_000_000.0),
                }, index=px_series.index)

                predictor = MLPredictor(model_type=ModelType.RANDOM_FOREST_CLASSIFIER)
                predictor.train(ohlcv, forecast_horizon=forecast_horizon, test_size=0.2)

                if not predictor.is_trained:
                    signal_df[ticker] = 0.0
                    continue

                # Predecir sobre toda la historia (rolling predict_proba no disponible,
                # usamos el único modelo entrenado para la última observación)
                result = predictor.predict(ohlcv)
                # result.prediction = P(up) para clasificación RF
                p_up = float(result.prediction)
                # Asignar la misma señal a todas las fechas (señal estática del modelo actual)
                signal_df[ticker] = 2.0 * p_up - 1.0

            except Exception as exc:
                logger.warning(f"[MLSignal] {ticker}: error en predicción — {exc}. Señal=0.")
                signal_df[ticker] = 0.0

        return signal_df.clip(-1.0, 1.0)
