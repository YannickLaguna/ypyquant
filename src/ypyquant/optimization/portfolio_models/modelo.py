"""
Modelo de portafolio compuesto: µ + Σ + optimizador.

Responsabilidad: `PortfolioOptimizationModel` encadena un `ExpectedReturnsModel`, un `RiskModel` y un
`PortfolioOptimizer` (pasa el histórico a los optimizadores que lo requieren y copia los avisos de los modelos
al resultado) y las fábricas `create_classic_markowitz_model`, `create_robust_model` y `create_capm_model`.

Se relaciona con: base.py, modelos_retorno.py, modelos_riesgo.py y optimizadores.py (las piezas),
optimizador_portafolio.py (`crear_modelo_personalizado` y el servicio lo construyen y ejecutan),
presets/ (modelos predefinidos) y walk_forward.py.
"""

import logging
from datetime import datetime

import pandas as pd

from ypyquant.optimization.portfolio_models.base import ExpectedReturnsModel, PortfolioOptimizer, RiskModel
from ypyquant.optimization.portfolio_models.modelos_retorno import CAPMReturnsModel, HistoricalReturnsModel
from ypyquant.optimization.portfolio_models.modelos_riesgo import LedoitWolfShrinkageModel
from ypyquant.optimization.portfolio_models.optimizadores import MarkowitzOptimizer, RiskParityOptimizer

logger = logging.getLogger(__name__)


class PortfolioOptimizationModel:
    """
    Clase principal para configurar y ejecutar optimización modular.
    Permite composición de diferentes componentes.

    TODO: Revisar lógica, no todo es compatible con todo
    """

    def __init__(
        self,
        name: str,
        returns_model: ExpectedReturnsModel,
        risk_model: RiskModel,
        optimizer: PortfolioOptimizer,
        description: str = ""
    ):
        self.name = name
        self.returns_model = returns_model
        self.risk_model = risk_model
        self.optimizer = optimizer
        self.description = description
        self.created_at = datetime.now()
        self.metadata = {}

    def optimize(
        self,
        returns_df: pd.DataFrame,
        constraints: dict | None = None,
        **kwargs
    ) -> dict:
        """
        Ejecuta optimización completa: calcula returns, risk y optimiza.

        Args:
            returns_df: DataFrame con retornos históricos
            constraints: Restricciones de optimización
            **kwargs: Parámetros adicionales para los modelos

        Returns:
            Dict con resultados completos de optimización
        """
        # Optimizadores que remuestrean el histórico (Michaud, CVaR) lo reciben de returns_df
        if getattr(self.optimizer, 'requiere_historico', False):
            kwargs.setdefault('retornos_historicos', returns_df)

        # 1. Calcular rendimientos esperados
        expected_returns = self.returns_model.calculate(returns_df, **kwargs)

        # 2. Calcular matriz de riesgo
        risk_matrix = self.risk_model.calculate(returns_df, **kwargs)

        # 3. Optimizar
        result = self.optimizer.optimize(
            expected_returns,
            risk_matrix,
            constraints,
            **kwargs
        )

        # Avisos del modelo de retornos (p. ej. BL sin market_caps) visibles en optimizer_metadata
        avisos = self.returns_model.metadata.get('avisos')
        if avisos:
            metadatos = result.setdefault('optimizer_metadata', {})
            # Se suman a los del optimizador (p. ej. volatilidad objetivo inalcanzable), no los sustituyen
            metadatos['avisos'] = list(avisos) + list(metadatos.get('avisos') or [])

        # Añadir metadata del modelo
        result['model_info'] = {
            'name': self.name,
            'description': self.description,
            'returns_model': self.returns_model.to_dict(),
            'risk_model': self.risk_model.to_dict(),
            'optimizer': self.optimizer.to_dict(),
            'created_at': self.created_at.isoformat(),
            'optimized_at': datetime.now().isoformat()
        }

        return result

    def to_dict(self) -> dict:
        """Serializa configuración completa del modelo"""
        return {
            'name': self.name,
            'description': self.description,
            'returns_model': self.returns_model.to_dict(),
            'risk_model': self.risk_model.to_dict(),
            'optimizer': self.optimizer.to_dict(),
            'created_at': self.created_at.isoformat()
        }

    def __repr__(self):
        return (
            f"PortfolioOptimizationModel("
            f"name='{self.name}', "
            f"returns={self.returns_model.name}, "
            f"risk={self.risk_model.name}, "
            f"optimizer={self.optimizer.name})"
        )


def create_classic_markowitz_model() -> PortfolioOptimizationModel:
    """Modelo clásico de Markowitz: CAPM Returns + Ledoit-Wolf Shrinkage + Max Sharpe"""
    return PortfolioOptimizationModel(
        name="Classic Markowitz",
        returns_model=CAPMReturnsModel(),
        risk_model=LedoitWolfShrinkageModel(),
        optimizer=MarkowitzOptimizer(objective="max_sharpe"),
        description="Modelo clásico de Markowitz con CAPM y shrinkage óptimo de Ledoit-Wolf"
    )


def create_robust_model() -> PortfolioOptimizationModel:
    """Modelo robusto: Median Returns + Ledoit-Wolf + Risk Parity"""
    return PortfolioOptimizationModel(
        name="Robust Risk Parity",
        returns_model=HistoricalReturnsModel(method="median"),
        risk_model=LedoitWolfShrinkageModel(),
        optimizer=RiskParityOptimizer(),
        description="Modelo robusto con shrinkage y risk parity"
    )


def create_capm_model() -> PortfolioOptimizationModel:
    """Modelo CAPM: CAPM Returns + Ledoit-Wolf Shrinkage + Max Sharpe"""
    return PortfolioOptimizationModel(
        name="CAPM Max Sharpe",
        returns_model=CAPMReturnsModel(),
        risk_model=LedoitWolfShrinkageModel(),
        optimizer=MarkowitzOptimizer(objective="max_sharpe"),
        description="Modelo basado en CAPM con shrinkage óptimo de Ledoit-Wolf"
    )
