"""
Predictor de retornos esperados usando LSTM (Long Short-Term Memory).
Usa secuencias temporales para predecir retornos futuros.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class LSTMReturnsPredictor:
    """
    Predictor de retornos esperados usando LSTM.

    Usa secuencias de retornos históricos para predecir retornos futuros
    de cada activo individualmente.

    NOTA: Requiere TensorFlow/Keras instalado:
        pip install tensorflow
    """

    def __init__(
        self,
        sequence_length: int = 60,
        forecast_horizon: int = 5,
        lstm_units: int = 50,
        dropout: float = 0.2
    ):
        """
        Args:
            sequence_length: Longitud de secuencia de entrada (días históricos)
            forecast_horizon: Horizonte de predicción (días futuros a promediar)
            lstm_units: Unidades en primera capa LSTM
            dropout: Dropout rate para regularización
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout = dropout

        self.models = {}  # Un modelo por ticker
        self.scalers = {}  # Scaler por ticker
        self.is_trained = False
        self.training_history = {}

    def _check_dependencies(self) -> bool:
        """Verifica que TensorFlow/Keras esté instalado"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            logger.info(f"[LSTM] TensorFlow {tf.__version__} disponible")
            return True
        except ImportError:
            logger.error(
                "[LSTM] TensorFlow no instalado. "
                "Instala con: pip install tensorflow"
            )
            return False

    def _create_sequences(
        self,
        data: np.ndarray,
        seq_length: int,
        forecast_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea secuencias de training para LSTM.

        Args:
            data: Array 1D con datos temporales
            seq_length: Longitud de secuencia de entrada
            forecast_horizon: Horizonte de predicción

        Returns:
            (X, y) donde:
            - X: secuencias de entrada shape (n_samples, seq_length)
            - y: targets (promedio de forecast_horizon días) shape (n_samples,)
        """
        X, y = [], []

        for i in range(len(data) - seq_length - forecast_horizon + 1):
            # Secuencia de entrada
            X.append(data[i:i+seq_length])

            # Target: promedio de los próximos forecast_horizon días
            y.append(data[i+seq_length:i+seq_length+forecast_horizon].mean())

        return np.array(X), np.array(y)

    def train(
        self,
        returns_df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 0
    ):
        """
        Entrena un modelo LSTM para cada ticker.

        Args:
            returns_df: DataFrame con retornos diarios (columnas = tickers)
            epochs: Número de épocas de training
            batch_size: Tamaño de batch
            validation_split: Fracción de datos para validación
            verbose: Nivel de verbosidad (0=silencioso, 1=barra progreso, 2=épocas)
        """
        if not self._check_dependencies():
            raise ImportError("TensorFlow requerido para LSTM. Ejecuta: pip install tensorflow")

        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler

        logger.info(
            f"[LSTM] Iniciando entrenamiento para {len(returns_df.columns)} activos\n"
            f"  - Sequence length: {self.sequence_length} días\n"
            f"  - Forecast horizon: {self.forecast_horizon} días\n"
            f"  - Epochs: {epochs}"
        )

        for ticker in returns_df.columns:
            logger.info(f"[LSTM] Entrenando modelo para {ticker}...")

            # Datos del ticker
            ticker_returns = returns_df[ticker].values

            # Normalizar (LSTM funciona mejor con datos normalizados)
            scaler = StandardScaler()
            scaled_returns = scaler.fit_transform(ticker_returns.reshape(-1, 1)).flatten()
            self.scalers[ticker] = scaler

            # Crear secuencias
            X, y = self._create_sequences(
                scaled_returns,
                self.sequence_length,
                self.forecast_horizon
            )

            if len(X) < 100:
                logger.warning(
                    f"[LSTM] Datos insuficientes para {ticker} ({len(X)} muestras), "
                    f"se requieren al menos 100. Saltando..."
                )
                continue

            # Reshape para LSTM: (samples, timesteps, features)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Crear modelo LSTM
            model = keras.Sequential([
                # Primera capa LSTM con return_sequences=True para stackear
                keras.layers.LSTM(
                    self.lstm_units,
                    activation='tanh',
                    return_sequences=True,
                    input_shape=(self.sequence_length, 1)
                ),
                keras.layers.Dropout(self.dropout),

                # Segunda capa LSTM
                keras.layers.LSTM(self.lstm_units // 2, activation='tanh'),
                keras.layers.Dropout(self.dropout),

                # Capa densa de salida
                keras.layers.Dense(1)
            ])

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )

            # Entrenar con early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            history = model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=verbose
            )

            self.models[ticker] = model
            self.training_history[ticker] = history.history

            # Métricas finales
            final_val_loss = history.history['val_loss'][-1]
            final_val_mae = history.history['val_mae'][-1]

            logger.info(
                f"[LSTM] ✓ {ticker} entrenado en {len(history.epoch)} épocas. "
                f"Val Loss: {final_val_loss:.6f}, Val MAE: {final_val_mae:.6f}"
            )

        self.is_trained = True
        logger.info(f"[LSTM] ✓ Entrenamiento completado para {len(self.models)}/{len(returns_df.columns)} modelos")

    def predict_expected_returns(
        self,
        returns_df: pd.DataFrame,
        annualize: bool = True
    ) -> pd.Series:
        """
        Predice retornos esperados usando modelos LSTM entrenados.

        Args:
            returns_df: Retornos históricos recientes (debe incluir al menos sequence_length días)
            annualize: Si True, anualiza los retornos predichos (* 252)

        Returns:
            Series con retornos esperados por ticker
        """
        if not self.is_trained:
            raise ValueError("Modelos no entrenados. Llama a train() primero.")

        import tensorflow as tf

        expected_returns = {}

        for ticker in returns_df.columns:
            if ticker not in self.models:
                logger.warning(
                    f"[LSTM] No hay modelo entrenado para {ticker}, usando fallback (media histórica)"
                )
                # Fallback: promedio histórico
                expected_returns[ticker] = returns_df[ticker].mean() * (252 if annualize else 1)
                continue

            # Datos recientes del ticker
            recent_returns = returns_df[ticker].values[-self.sequence_length:]

            if len(recent_returns) < self.sequence_length:
                logger.warning(
                    f"[LSTM] Datos insuficientes para {ticker} "
                    f"({len(recent_returns)} < {self.sequence_length}), usando fallback"
                )
                expected_returns[ticker] = returns_df[ticker].mean() * (252 if annualize else 1)
                continue

            # Normalizar con el mismo scaler de training
            scaler = self.scalers[ticker]
            scaled_recent = scaler.transform(recent_returns.reshape(-1, 1)).flatten()

            # Reshape para predicción: (1, sequence_length, 1)
            X = scaled_recent.reshape((1, self.sequence_length, 1))

            # Predecir
            model = self.models[ticker]
            prediction_scaled = model.predict(X, verbose=0)[0][0]

            # Denormalizar
            prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]

            # Anualizar si se requiere
            if annualize:
                prediction = prediction * 252

            expected_returns[ticker] = float(prediction)

        return pd.Series(expected_returns)

    def get_training_summary(self) -> pd.DataFrame:
        """
        Retorna resumen del entrenamiento de todos los modelos.

        Returns:
            DataFrame con métricas de training por ticker
        """
        if not self.is_trained:
            return pd.DataFrame()

        summary = []

        for ticker, history in self.training_history.items():
            summary.append({
                'ticker': ticker,
                'epochs_trained': len(history['loss']),
                'final_train_loss': history['loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_train_mae': history['mae'][-1],
                'final_val_mae': history['val_mae'][-1],
                'best_val_loss': min(history['val_loss'])
            })

        return pd.DataFrame(summary)

    def save_models(self, directory: str):
        """
        Guarda modelos entrenados a disco.

        Args:
            directory: Directorio donde guardar los modelos
        """
        if not self.is_trained:
            raise ValueError("No hay modelos entrenados para guardar")

        import os
        import pickle

        os.makedirs(directory, exist_ok=True)

        # Guardar cada modelo LSTM
        for ticker, model in self.models.items():
            model_path = os.path.join(directory, f"{ticker}_lstm.h5")
            model.save(model_path)

        # Guardar scalers
        scalers_path = os.path.join(directory, "scalers.pkl")
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)

        # Guardar metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'tickers': list(self.models.keys())
        }

        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"[LSTM] ✓ Modelos guardados en {directory}/")

    def load_models(self, directory: str):
        """
        Carga modelos previamente entrenados desde disco.

        Args:
            directory: Directorio de donde cargar los modelos
        """
        import os
        import pickle
        from tensorflow import keras

        # Cargar metadata
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.sequence_length = metadata['sequence_length']
        self.forecast_horizon = metadata['forecast_horizon']
        self.lstm_units = metadata['lstm_units']
        self.dropout = metadata['dropout']

        # Cargar scalers
        scalers_path = os.path.join(directory, "scalers.pkl")
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)

        # Cargar modelos LSTM
        for ticker in metadata['tickers']:
            model_path = os.path.join(directory, f"{ticker}_lstm.h5")
            if os.path.exists(model_path):
                self.models[ticker] = keras.models.load_model(model_path)

        self.is_trained = True
        logger.info(f"[LSTM] ✓ Modelos cargados desde {directory}/ ({len(self.models)} tickers)")

    def to_dict(self) -> Dict:
        """Serializa información del predictor"""
        result = {
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'is_trained': self.is_trained,
            'n_models': len(self.models),
            'tickers': list(self.models.keys()) if self.is_trained else []
        }

        return result
