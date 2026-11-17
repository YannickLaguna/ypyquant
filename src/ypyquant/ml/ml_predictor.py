"""
ML Predictor - Predicción de movimientos con Machine Learning.
Framework esquemático extensible para modelos de ML.

Soporta:
- Clasificación (dirección del movimiento)
- Regresión (magnitud del retorno)
- Feature engineering automático
- Múltiples modelos (XGBoost, LightGBM, RandomForest)
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipos de modelos soportados"""
    XGBOOST_CLASSIFIER = "xgboost_classifier"
    XGBOOST_REGRESSOR = "xgboost_regressor"
    LIGHTGBM_CLASSIFIER = "lightgbm_classifier"
    LIGHTGBM_REGRESSOR = "lightgbm_regressor"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"


@dataclass
class PredictionResult:
    """Resultado de una predicción"""
    ticker: str
    prediction: float  # Probabilidad o valor predicho
    confidence: float  # Nivel de confianza
    features_used: list[str]
    model_type: str


class FeatureEngineering:
    """
    Generador de features para ML.

    Crea features técnicos a partir de datos de precio.
    """

    @staticmethod
    def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features técnicos.

        Args:
            df: DataFrame con OHLCV data

        Returns:
            DataFrame con features adicionales
        """
        features = df.copy()

        # Returns
        features['returns'] = features['Close'].pct_change()
        features['returns_5d'] = features['Close'].pct_change(5)
        features['returns_10d'] = features['Close'].pct_change(10)
        features['returns_20d'] = features['Close'].pct_change(20)

        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            features[f'sma_{window}'] = features['Close'].rolling(window).mean()
            features[f'ema_{window}'] = features['Close'].ewm(span=window).mean()

        # Price position
        features['price_to_sma_20'] = features['Close'] / features['sma_20']
        features['price_to_sma_50'] = features['Close'] / features['sma_50']

        # Volatility
        features['volatility_10d'] = features['returns'].rolling(10).std()
        features['volatility_20d'] = features['returns'].rolling(20).std()

        # Volume
        features['volume_sma_20'] = features['Volume'].rolling(20).mean()
        features['volume_ratio'] = features['Volume'] / features['volume_sma_20']

        # Momentum indicators
        features['rsi_14'] = FeatureEngineering._calculate_rsi(features['Close'], 14)
        features['macd'], features['macd_signal'] = FeatureEngineering._calculate_macd(features['Close'])

        # Bollinger Bands
        bb_upper, bb_lower = FeatureEngineering._calculate_bollinger_bands(features['Close'])
        features['bb_position'] = (features['Close'] - bb_lower) / (bb_upper - bb_lower)

        return features

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def _calculate_macd(prices: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Calcula MACD."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()

        return macd, signal

    @staticmethod
    def _calculate_bollinger_bands(
        prices: pd.Series,
        window: int = 20,
        num_std: int = 2
    ) -> tuple[pd.Series, pd.Series]:
        """Calcula Bollinger Bands."""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()

        upper = sma + (std * num_std)
        lower = sma - (std * num_std)

        return upper, lower

    @staticmethod
    def create_labels(
        df: pd.DataFrame,
        forecast_horizon: int = 5,
        task: str = 'classification'
    ) -> pd.Series:
        """
        Crea labels para training.

        Args:
            df: DataFrame con datos
            forecast_horizon: Horizonte de predicción (días)
            task: 'classification' o 'regression'

        Returns:
            Series con labels
        """
        future_returns = df['Close'].shift(-forecast_horizon).pct_change(forecast_horizon)

        if task == 'classification':
            # Label: 1 si sube, 0 si baja
            labels = (future_returns > 0).astype(int)
        else:  # regression
            # Label: retorno futuro
            labels = future_returns

        return labels


class MLPredictor:
    """
    Predictor de movimientos usando Machine Learning.

    Framework esquemático para integrar modelos de ML.
    """

    def __init__(self, model_type: ModelType = ModelType.RANDOM_FOREST_CLASSIFIER):
        """
        Inicializa el predictor.

        Args:
            model_type: Tipo de modelo a usar
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.is_trained = False

        logger.info(f"MLPredictor inicializado con {model_type.value}")

    def train(
        self,
        df: pd.DataFrame,
        forecast_horizon: int = 5,
        test_size: float = 0.2
    ) -> dict:
        """
        Entrena el modelo.

        Args:
            df: DataFrame con datos OHLCV
            forecast_horizon: Horizonte de predicción
            test_size: Tamaño del conjunto de test

        Returns:
            Dict con métricas de training
        """
        logger.info(f"Iniciando training con {len(df)} muestras")

        # Feature engineering
        features_df = FeatureEngineering.create_technical_features(df)

        # Crear labels
        task = 'classification' if 'classifier' in self.model_type.value else 'regression'
        labels = FeatureEngineering.create_labels(features_df, forecast_horizon, task)

        # Alinear features y labels
        features_df = features_df.iloc[:-forecast_horizon]  # Remover últimas filas sin label
        labels = labels.iloc[:-forecast_horizon]

        # Dropear NaN
        valid_idx = features_df.notna().all(axis=1) & labels.notna()
        features_df = features_df[valid_idx]
        labels = labels[valid_idx]

        if len(features_df) == 0:
            raise ValueError("No hay datos válidos después de feature engineering")

        # Seleccionar features numéricas
        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        X = features_df[numeric_features]
        y = labels

        self.feature_names = numeric_features

        # Split train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Entrenar modelo (usando sklearn como ejemplo)
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

        if task == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)

            # Evaluar
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            metrics = {
                'accuracy': accuracy,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(self.feature_names)
            }

            logger.info(f"Training completado: Accuracy={accuracy:.2%}")

        else:  # regression
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)

            # Evaluar
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(self.feature_names)
            }

            logger.info(f"Training completado: RMSE={np.sqrt(mse):.4f}, R2={r2:.4f}")

        self.is_trained = True
        return metrics

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """
        Hace predicción sobre datos nuevos.

        Args:
            df: DataFrame con datos OHLCV recientes

        Returns:
            PredictionResult con predicción
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")

        # Feature engineering
        features_df = FeatureEngineering.create_technical_features(df)

        # Usar última fila (datos más recientes)
        X = features_df[self.feature_names].iloc[[-1]]

        # Verificar NaN
        if X.isna().any().any():
            logger.warning("Features contienen NaN, reemplazando con 0")
            X = X.fillna(0)

        # Predicción
        prediction = self.model.predict(X)[0]

        # Confidence (probabilidades para clasificación)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = max(proba)
        else:
            confidence = 0.5  # Default para regresión

        ticker = df.index.name if df.index.name else "Unknown"

        return PredictionResult(
            ticker=ticker,
            prediction=float(prediction),
            confidence=confidence,
            features_used=self.feature_names,
            model_type=self.model_type.value
        )

    def get_feature_importance(self) -> dict[str, float]:
        """
        Obtiene importancia de features.

        Returns:
            Dict con importancia por feature
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}

        importances = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importances))

        # Ordenar por importancia
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return dict(sorted_features)


def format_prediction_result(result: PredictionResult) -> str:
    """Formatea resultado de predicción."""
    direction = "UP ↑" if result.prediction > 0.5 else "DOWN ↓"

    return f"""
Prediction for {result.ticker}:
  Direction:     {direction}
  Probability:   {result.prediction:.1%}
  Confidence:    {result.confidence:.1%}
  Model:         {result.model_type}
  Features:      {len(result.features_used)}
    """.strip()
