"""
Presets Avanzados con Machine Learning
======================================

Modelos que utilizan técnicas de ML para mejorar la optimización:
- ML-Enhanced Markowitz (Clustering + PCA)
- Regime-Aware Adaptive (SVM)
- LSTM Predictive Portfolio

NOTA: Estos presets requieren dependencias adicionales:
- scikit-learn (clustering, PCA, SVM)
- tensorflow (LSTM)
"""

from quant.optimization.presets import registrar_preset
from typing import Dict


@registrar_preset('ml_enhanced')
def crear_ml_enhanced_markowitz(risk_free_rate: float = 0.02) -> Dict:
    """
    Markowitz mejorado con Machine Learning y Ledoit-Wolf Shrinkage.

    Pipeline end-to-end:
    1. Pre-filtrado: Clustering jerárquico para reducir universo
       - Agrupa activos por correlación
       - Selecciona representante de cada cluster (mejor Sharpe)
       - Reduce de N activos a ~10-15 representantes

    2. Retornos: Media histórica

    3. Riesgo: Ledoit-Wolf shrinkage (óptimo teóricamente)
       - Matriz de covarianza estabilizada
       - Shrinkage óptimo automático
       - Bien condicionada numéricamente

    4. Optimización: Maximizar Sharpe

    Ventajas:
    - Mucho más estable que Markowitz clásico
    - Maneja portfolios grandes (100+ activos)
    - Reduce overfitting con clustering + shrinkage óptimo
    - Matriz de covarianza robusta

    Requisitos:
    - scikit-learn>=1.0

    Args:
        risk_free_rate: Tasa libre de riesgo para cálculos (default: 0.02)
    """
    return {
        'nombre': 'ML-Enhanced Markowitz con Ledoit-Wolf',
        'pipeline': {
            'preprocesamiento': {
                'usar_clustering': True,
                'metodo_clustering': 'jerarquico',
                'n_clusters': 10,
                'metodo_seleccion': 'max_sharpe'
            },
            'tipo_modelo_retornos': 'media_historica',
            'config_retornos': {'frecuencia': 252},
            'tipo_modelo_riesgo': 'ledoit_wolf',
            'config_riesgo': {'frecuencia': 252},
            'tipo_optimizador': 'max_sharpe',
            'config_optimizador': {'tasa_libre_riesgo': risk_free_rate}
        },
        'description': 'Markowitz con clustering + Ledoit-Wolf shrinkage para máxima estabilidad',
        'recomendado_para': 'Portfolios grandes (50-200 activos)',
        'complejidad': 'Alta',
        'tiempo_computo': 'Media',
        'requisitos': ['scikit-learn>=1.0']
    }


@registrar_preset('regime_aware')
def crear_regime_aware_adaptive(risk_free_rate: float = 0.02) -> Dict:
    """
    Portfolio adaptativo según régimen de mercado detectado con SVM.

    Pipeline end-to-end:
    1. Detección de Régimen: SVM clasifica mercado actual
       - Bear Market (bajista)
       - Bull Market (alcista)
       - Lateral/Consolidación
       - High Volatility

    2. Ajuste Dinámico: Cambia parámetros según régimen (TODOS CON LEDOIT-WOLF)
       - Bear → Min Volatility + Ledoit-Wolf + Max 25% por activo
       - Bull → Max Sharpe + Ledoit-Wolf + Max 40% por activo
       - Lateral → Max Sharpe + Ledoit-Wolf + Max 30% por activo
       - High Vol → Risk Parity + Ledoit-Wolf + Max 20% por activo

    3. Cash Allocation: Variable según régimen
       - Bear: 30% cash
       - Bull: 5% cash
       - Lateral: 10% cash
       - High Vol: 25% cash

    Ventajas:
    - Se adapta automáticamente a condiciones de mercado
    - Protección en bear markets
    - Agresivo en bull markets
    - Matriz de covarianza estable en todos los régimenes (Ledoit-Wolf)
    - No requiere intervención manual

    Requisitos:
    - scikit-learn>=1.0 (SVM)
    """
    return {
        'nombre': 'Regime-Aware Adaptive',
        'pipeline': {
            'deteccion_regimen': {
                'habilitado': True,
                'tipo_modelo': 'svm',
                'ticker_mercado': '^GSPC',
                'ventana_historica': 60
            },
            'ajuste_dinamico': True,
            'tipo_modelo_retornos': 'media_historica',  # Base, se ajusta según régimen
            'tipo_modelo_riesgo': 'dinamico',  # Cambia según régimen
            'tipo_optimizador': 'dinamico',  # Cambia según régimen
            'asignacion_efectivo': 'dinamica'  # Cambia según régimen
        },
        'description': 'Portfolio que se adapta automáticamente al régimen de mercado (SVM)',
        'recomendado_para': 'Mercados volátiles, inversores activos',
        'complejidad': 'Muy Alta',
        'tiempo_computo': 'Alta (requiere entrenar SVM)',
        'requisitos': ['scikit-learn>=1.0']
    }


@registrar_preset('lstm_predictivo')
def crear_lstm_predictive_portfolio(risk_free_rate: float = 0.02) -> Dict:
    """
    Portfolio con predicciones de retornos usando LSTM.

    Pipeline end-to-end:
    1. Predicción LSTM: Entrena modelo LSTM para cada activo
       - Usa 60 días históricos para predecir próximos 5 días
       - Modelos individuales por ticker
       - Regularización con dropout

    2. Retornos Esperados: Predicciones LSTM (en lugar de medias históricas)

    3. Riesgo: Ledoit-Wolf shrinkage (robusta)

    4. Optimización: Maximizar Sharpe

    Ventajas:
    - Captura patrones temporales complejos
    - Mejor que medias simples en mercados con tendencias
    - Adaptativo a cambios de régimen

    Desventajas:
    - Requiere TensorFlow (pesado)
    - Tiempo de entrenamiento alto
    - Puede overfittear con pocos datos

    Requisitos:
    - tensorflow>=2.15.0

    Args:
        risk_free_rate: Tasa libre de riesgo para cálculos (default: 0.02)
    """
    return {
        'nombre': 'LSTM Predictive Portfolio',
        'pipeline': {
            'tipo_modelo_retornos': 'lstm',
            'config_retornos': {
                'longitud_secuencia': 60,
                'horizonte_prediccion': 5,
                'unidades_lstm': 50,
                'dropout': 0.2,
                'epochs': 50,
                'batch_size': 32
            },
            'tipo_modelo_riesgo': 'ledoit_wolf',
            'config_riesgo': {'frecuencia': 252},
            'tipo_optimizador': 'max_sharpe',
            'config_optimizador': {'tasa_libre_riesgo': risk_free_rate}
        },
        'description': 'Portfolio con predicciones de retornos usando redes LSTM',
        'recomendado_para': 'Inversores cuantitativos, mercados con tendencias',
        'complejidad': 'Muy Alta',
        'tiempo_computo': 'Muy Alta (30-60 min para 50 activos)',
        'requisitos': ['tensorflow>=2.15.0']
    }
