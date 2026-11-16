"""
Clasificador de régimen de mercado usando SVM o Random Forest.
Detecta: Bull/Bear/Lateral/High Volatility y ajusta parámetros de optimización.
"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """
    Clasifica régimen de mercado usando machine learning.

    Regímenes detectados:
    - 0: Bear Market (bajista)
    - 1: Lateral/Consolidation (consolidación)
    - 2: Bull Market (alcista)
    - 3: High Volatility (alta volatilidad)
    """

    REGIME_NAMES = {
        0: 'Bear Market',
        1: 'Lateral/Consolidation',
        2: 'Bull Market',
        3: 'High Volatility'
    }

    def __init__(self, model_type: str = 'svm'):
        """
        Args:
            model_type: 'svm' o 'random_forest'
        """
        self.model_type = model_type

        if model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale')
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"model_type desconocido: {model_type}")

        self.is_trained = False
        self.feature_importance = None

    def create_regime_labels(
        self,
        market_returns: pd.Series,
        lookback: int = 60
    ) -> pd.Series:
        """
        Crea labels de régimen basados en características del mercado.

        Criterios:
        - Bear: Retorno promedio < -5% anual Y tendencia negativa
        - Bull: Retorno promedio > 10% anual Y tendencia positiva
        - High Vol: Volatilidad > 25% anual
        - Lateral: Todo lo demás

        Args:
            market_returns: Series con retornos diarios del mercado
            lookback: Ventana para calcular métricas

        Returns:
            Series con labels de régimen
        """
        labels = []

        for i in range(lookback, len(market_returns)):
            window = market_returns.iloc[i-lookback:i]

            # Métricas
            mean_ret = window.mean() * 252  # Anualizado
            volatility = window.std() * np.sqrt(252)  # Anualizada
            trend = np.polyfit(range(len(window)), window.values, 1)[0] * 252  # Pendiente anualizada

            # Clasificar régimen
            if volatility > 0.25:
                regime = 3  # High Volatility
            elif mean_ret < -0.05 and trend < 0:
                regime = 0  # Bear
            elif mean_ret > 0.10 and trend > 0:
                regime = 2  # Bull
            else:
                regime = 1  # Lateral

            labels.append(regime)

        return pd.Series(labels, index=market_returns.index[lookback:])

    def create_features(
        self,
        market_returns: pd.Series,
        lookback: int = 60
    ) -> pd.DataFrame:
        """
        Crea features técnicos para clasificación de régimen.

        Features calculados:
        - mean_return: Retorno promedio anualizado
        - volatility: Volatilidad anualizada
        - trend: Tendencia (pendiente regresión lineal)
        - momentum: Retorno acumulado
        - rsi: Relative Strength Index
        - max_drawdown: Máximo drawdown
        - skewness: Asimetría de distribución
        - kurtosis: Curtosis (fat tails)

        Args:
            market_returns: Retornos diarios
            lookback: Ventana rolling

        Returns:
            DataFrame con features
        """
        features = []

        for i in range(lookback, len(market_returns)):
            window = market_returns.iloc[i-lookback:i]

            # 1. Retorno promedio anualizado
            mean_ret = window.mean() * 252

            # 2. Volatilidad anualizada
            volatility = window.std() * np.sqrt(252)

            # 3. Tendencia (pendiente regresión lineal)
            trend = np.polyfit(range(len(window)), window.values, 1)[0] * 252

            # 4. Momentum (retorno acumulado)
            momentum = (1 + window).prod() - 1

            # 5. RSI (Relative Strength Index)
            delta = window.diff()
            gain = delta.where(delta > 0, 0).mean()
            loss = -delta.where(delta < 0, 0).mean()
            rs = gain / loss if loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))

            # 6. Max Drawdown
            cumulative = (1 + window).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()

            # 7. Skewness (asimetría)
            skewness = window.skew()

            # 8. Kurtosis (fat tails)
            kurtosis = window.kurtosis()

            features.append({
                'mean_return': mean_ret,
                'volatility': volatility,
                'trend': trend,
                'momentum': momentum,
                'rsi': rsi,
                'max_drawdown': max_dd,
                'skewness': skewness,
                'kurtosis': kurtosis
            })

        return pd.DataFrame(features, index=market_returns.index[lookback:])

    def train(
        self,
        market_returns: pd.Series,
        lookback: int = 60
    ):
        """
        Entrena el clasificador de régimen.

        Args:
            market_returns: Series con retornos diarios del mercado (ej: S&P 500)
            lookback: Ventana para features
        """
        logger.info(f"[MarketRegime] Entrenando modelo {self.model_type}...")

        # Crear features y labels
        X = self.create_features(market_returns, lookback)
        y = self.create_regime_labels(market_returns, lookback)

        # Alinear índices
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        if len(X) < 100:
            raise ValueError(
                f"Datos insuficientes para entrenar. Necesitas al menos 100 muestras, "
                f"tienes {len(X)}. Aumenta el rango de fechas."
            )

        # Entrenar
        self.model.fit(X, y)
        self.is_trained = True

        # Calcular feature importance (solo para RF)
        if self.model_type == 'random_forest':
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)

        # Métricas de entrenamiento
        train_accuracy = self.model.score(X, y)

        logger.info(f"[MarketRegime] ✓ Modelo entrenado con {len(X)} muestras")
        logger.info(f"[MarketRegime] Accuracy en training: {train_accuracy:.2%}")
        logger.info(f"[MarketRegime] Distribución de regímenes:")

        regime_dist = y.value_counts(normalize=True).sort_index()
        for regime_id, pct in regime_dist.items():
            regime_name = self.REGIME_NAMES.get(regime_id, 'Unknown')
            logger.info(f"  {regime_name}: {pct:.1%}")

        if self.feature_importance is not None:
            logger.info(f"[MarketRegime] Top 3 features más importantes:")
            for feature, importance in self.feature_importance.head(3).items():
                logger.info(f"  {feature}: {importance:.3f}")

    def predict_current_regime(
        self,
        market_returns: pd.Series,
        lookback: int = 60
    ) -> Tuple[int, Dict[int, float]]:
        """
        Predice el régimen actual del mercado.

        Args:
            market_returns: Retornos diarios del mercado
            lookback: Ventana para features

        Returns:
            (regime_id, probabilities_dict)
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")

        # Features de los últimos días
        features = self.create_features(market_returns, lookback)

        if len(features) == 0:
            raise ValueError(
                f"No se pudieron calcular features. "
                f"Necesitas al menos {lookback} días de datos."
            )

        X_current = features.iloc[[-1]]

        # Predicción
        regime = self.model.predict(X_current)[0]
        probabilities = self.model.predict_proba(X_current)[0]

        # Convertir a dict
        prob_dict = {i: float(p) for i, p in enumerate(probabilities)}

        regime_name = self.REGIME_NAMES.get(regime, 'Unknown')
        logger.info(
            f"[MarketRegime] Régimen detectado: {regime_name} "
            f"(confianza: {probabilities[regime]:.1%})"
        )

        return int(regime), prob_dict

    @staticmethod
    def get_regime_name(regime_id: int) -> str:
        """Retorna nombre legible del régimen"""
        return MarketRegimeClassifier.REGIME_NAMES.get(regime_id, 'Unknown')

    @staticmethod
    def get_optimization_params_for_regime(regime_id: int, risk_free_rate: float = 0.02) -> Dict:
        """
        Retorna parámetros de optimización recomendados por régimen.

        Args:
            regime_id: ID del régimen (0-3)
            risk_free_rate: Tasa libre de riesgo

        Returns:
            Dict con configuración de optimización
        """
        if regime_id == 0:  # Bear Market
            return {
                'objective': 'min_volatility',
                'constraints': {'max_weight': 0.25},
                'risk_model': 'semi_covariance',
                'cash_allocation': 0.30,  # Más cash en bear
                'risk_free_rate': risk_free_rate,
                'description': 'Estrategia defensiva para mercado bajista'
            }

        elif regime_id == 1:  # Lateral
            return {
                'objective': 'max_sharpe',
                'constraints': {'max_weight': 0.30},
                'risk_model': 'ledoit_wolf',
                'cash_allocation': 0.10,
                'risk_free_rate': risk_free_rate,
                'description': 'Estrategia balanceada para consolidación'
            }

        elif regime_id == 2:  # Bull Market
            return {
                'objective': 'max_sharpe',
                'constraints': {'max_weight': 0.40},
                'risk_model': 'sample_covariance',
                'cash_allocation': 0.05,  # Menos cash en bull
                'risk_free_rate': risk_free_rate,
                'description': 'Estrategia agresiva para mercado alcista'
            }

        else:  # High Volatility
            return {
                'objective': 'risk_parity',
                'constraints': {'max_weight': 0.20},
                'risk_model': 'exponential_weighted',
                'cash_allocation': 0.25,
                'risk_free_rate': risk_free_rate,
                'description': 'Risk parity para alta volatilidad'
            }

    def to_dict(self) -> Dict:
        """Serializa información del modelo"""
        result = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'regime_names': self.REGIME_NAMES
        }

        if self.feature_importance is not None:
            result['feature_importance'] = self.feature_importance.to_dict()

        return result
