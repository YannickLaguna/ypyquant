"""
Presets Basados en Modelos de Factores
======================================

Modelos que usan factores de riesgo para explicar retornos:
- CAPM (Capital Asset Pricing Model)
- Black-Litterman
- Fama-French 5 Factores 

Estos modelos incorporan teoría financiera sobre factores de riesgo sistemático.
"""

from ypyquant.optimization.portfolio_models import (
    PortfolioOptimizationModel,
    CAPMReturnsModel,
    BlackLittermanReturnsModel,
    LedoitWolfShrinkageModel,
    MarkowitzOptimizer
)
from ypyquant.optimization.presets import registrar_preset
from typing import Dict


@registrar_preset('capm')
def crear_capm_max_sharpe(risk_free_rate: float = 0.02) -> PortfolioOptimizationModel:
    """
    Modelo basado en CAPM con Ledoit-Wolf Shrinkage.

    Pipeline:
    - Retornos: CAPM (E[R] = Rf + β*(Rm - Rf))
    - Riesgo: Ledoit-Wolf shrinkage (matriz estable y robusta)
    - Optimización: Maximizar Sharpe Ratio

    Características:
    - Basado en teoría financiera sólida (CAPM)
    - Matriz de covarianza estabilizada con shrinkage óptimo
    - Requiere datos de mercado (ej: S&P 500)
    - Betas estables en el tiempo

    Casos de uso:
    - Portfolios de acciones
    - Cuando hay buen benchmark de mercado
    - Análisis factor-based con estabilidad numérica

    Args:
        risk_free_rate: Tasa libre de riesgo para CAPM y Sharpe (default: 0.02)
    """
    return PortfolioOptimizationModel(
        name="CAPM con Ledoit-Wolf",
        returns_model=CAPMReturnsModel(market_ticker="^GSPC", risk_free_rate=risk_free_rate),
        risk_model=LedoitWolfShrinkageModel(frequency=252),
        optimizer=MarkowitzOptimizer(objective="max_sharpe", risk_free_rate=risk_free_rate),
        description="Modelo CAPM con shrinkage óptimo de Ledoit-Wolf"
    )


@registrar_preset('black_litterman')
def crear_black_litterman_mejorado(risk_free_rate: float = 0.02) -> PortfolioOptimizationModel:
    """
    Black-Litterman con equilibrio de mercado.

    Pipeline:
    1. Prior: Equilibrio de mercado basado en market caps
       Π = δ * Σ * w_mkt (implied returns)

    2. Views: Opcionales (pueden agregarse manualmente)
       - Vistas absolutas: "AAPL retornará 15% este año"
       - Vistas relativas: "GOOGL superará a META por 5%"

    3. Posterior: Combinación bayesiana de prior + views

    4. Riesgo: Ledoit-Wolf shrinkage

    5. Optimización: Maximizar Sharpe

    Ventajas:
    - Combina equilibrio de mercado con opiniones del inversor
    - Más estable que medias históricas puras
    - Framework bayesiano riguroso

    Casos de uso:
    - Inversores con views de mercado específicas
    - Portfolios institucionales
    - Gestión activa con opiniones

    Args:
        risk_free_rate: Tasa libre de riesgo para cálculos (default: 0.02)
    """
    return PortfolioOptimizationModel(
        name="Black-Litterman Mejorado",
        returns_model=BlackLittermanReturnsModel(
            risk_aversion=2.5,
            tau=0.025,
            market_caps=None  # Se descarga automáticamente
        ),
        risk_model=LedoitWolfShrinkageModel(frequency=252),
        optimizer=MarkowitzOptimizer(objective="max_sharpe", risk_free_rate=risk_free_rate),
        description="Black-Litterman con equilibrio de mercado y views opcionales"
    )


@registrar_preset('fama_french_4')
def crear_fama_french_4_factores(risk_free_rate: float = 0.02) -> Dict:
    """
    Modelo Fama-French 4 factores con datos REALES y Ledoit-Wolf Shrinkage.

    Pipeline:
    1. Retornos: Fama-French 4-Factor Model
       - MKT (Market premium)
       - SMB (Small Minus Big - factor tamaño)
       - HML (High Minus Low - factor valor)
       - UMD (Up Minus Down - momentum)
       - Descarga factores REALES desde Kenneth French Data Library

    2. Riesgo: Ledoit-Wolf shrinkage (matriz estable y robusta)

    3. Optimización: Maximizar Sharpe

    Ventajas:
    - Basado en investigación académica sólida (Fama & French)
    - Explica retornos mejor que CAPM
    - Factores reales, no simulados
    - Matriz de covarianza óptimamente estabilizada

    NOTA: Requiere librería famafrench o pandas-datareader

    Args:
        risk_free_rate: Tasa libre de riesgo para cálculos (default: 0.02)
    """
    return {
        'nombre': 'Fama-French 4-Factor con Ledoit-Wolf',
        'pipeline': {
            'tipo_modelo_retornos': 'fama_french_4',
            'config_retornos': {
                'tasa_libre_riesgo': risk_free_rate,
                'fuente_datos': 'kenneth_french'  # Descarga automática
            },
            'tipo_modelo_riesgo': 'ledoit_wolf',
            'config_riesgo': {'frecuencia': 252},
            'tipo_optimizador': 'max_sharpe',
            'config_optimizador': {'tasa_libre_riesgo': risk_free_rate}
        },
        'description': 'Modelo Fama-French 4 factores con shrinkage óptimo de Ledoit-Wolf',
        'recomendado_para': 'Análisis factor-based, portfolios de acciones',
        'complejidad': 'Media',
        'tiempo_computo': 'Media',
        'requisitos': ['famafrench>=0.1.4 o pandas-datareader>=0.10.0']
    }
