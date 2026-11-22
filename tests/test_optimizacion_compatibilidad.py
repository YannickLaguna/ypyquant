"""
test_optimizacion_compatibilidad.py — Contrato de compatibilidad de la API pública de optimización.

Responsabilidad:
    - Comprobar que `ypyquant.optimization.portfolio_models` (ahora un subpaquete partido por tipo)
      sigue exponiendo, con la misma ruta, todos los nombres públicos que tenía el módulo único, y que son
      los mismos objetos que los definidos en los módulos internos.
    - Fijar la superficie de `PortfolioOptimizationService` que GrowSave subclasifica o llama (constructor,
      ganchos, métodos) y las funciones y presets que GrowSave importa.

Se relaciona con: optimization/portfolio_models/, optimization/optimizador_portafolio.py,
optimization/presets/ y docs/planes/2026-11-21-optimizacion-canonica.md.
"""

import importlib
import inspect

import pytest

NOMBRES_PORTFOLIO_MODELS = [
    "ReturnModelType", "RiskModelType", "OptimizerType", "ExpectedReturnsModel", "RiskModel",
    "PortfolioOptimizer", "HistoricalReturnsModel", "CAPMReturnsModel", "BlackLittermanReturnsModel",
    "LSTMReturnsModel", "BayesSteinReturnsModel", "BLPriorReturnsModel", "BLMomentumReturnsModel",
    "SampleCovarianceModel", "LedoitWolfShrinkageModel", "ExponentialCovarianceModel", "SemiCovarianceModel",
    "LedoitWolfSemiCovarianceModel", "PCADenoisedRiskModel", "MarkowitzOptimizer", "RiskParityOptimizer",
    "PMPTMinCVaROptimizer", "PMPTMaxSortinoOptimizer", "MaxSharpeQPOptimizer", "MichaudOptimizer",
    "RobustoElipsoideOptimizer", "RobustoCajaOptimizer", "PortfolioOptimizationModel",
    "create_classic_markowitz_model", "create_robust_model", "create_capm_model",
]
MODULOS_INTERNOS = ["base", "modelos_retorno", "modelos_riesgo", "optimizadores", "modelo"]


@pytest.mark.parametrize("nombre", NOMBRES_PORTFOLIO_MODELS)
def test_portfolio_models_reexporta_todos_los_nombres_publicos(nombre):
    paquete = importlib.import_module("ypyquant.optimization.portfolio_models")
    assert hasattr(paquete, nombre), nombre
    definido_en = [
        m for m in MODULOS_INTERNOS
        if getattr(importlib.import_module(f"ypyquant.optimization.portfolio_models.{m}"), nombre, None)
        is getattr(paquete, nombre)
    ]
    assert definido_en, f"{nombre} no coincide con ningún módulo interno"


def test_all_del_paquete_cubre_los_nombres_antiguos():
    paquete = importlib.import_module("ypyquant.optimization.portfolio_models")
    assert set(NOMBRES_PORTFOLIO_MODELS) <= set(paquete.__all__)


def test_shim_de_growsave_importa_todo():
    """Lista literal de `app/services/optimizacion/portfolio_models.py` de GrowSave."""
    from ypyquant.optimization.portfolio_models import (  # noqa: F401
        BlackLittermanReturnsModel,
        CAPMReturnsModel,
        ExpectedReturnsModel,
        ExponentialCovarianceModel,
        HistoricalReturnsModel,
        LedoitWolfSemiCovarianceModel,
        LedoitWolfShrinkageModel,
        MarkowitzOptimizer,
        OptimizerType,
        PMPTMaxSortinoOptimizer,
        PMPTMinCVaROptimizer,
        PortfolioOptimizationModel,
        PortfolioOptimizer,
        ReturnModelType,
        RiskModel,
        RiskModelType,
        RiskParityOptimizer,
        SampleCovarianceModel,
        SemiCovarianceModel,
        create_capm_model,
        create_classic_markowitz_model,
        create_robust_model,
    )


def test_superficie_del_servicio_que_usa_growsave():
    from ypyquant.optimization.optimizador_portafolio import PortfolioOptimizationService as Servicio

    parametros = inspect.signature(Servicio.__init__).parameters
    assert {"cargador_precios", "tipo_retorno"} <= set(parametros)
    metodos = [
        "_get_returns_matrix", "_obtener_datos_modelo", "optimize_advanced", "compare_models",
        "calcular_metricas_cartera", "calculate_efficient_frontier", "optimize_portfolio",
        "adjust_portfolio_to_volatility", "create_custom_model", "get_available_models",
        "get_preconfigured_model", "list_available_presets", "get_preset_details", "optimize_target_volatility",
        "optimize_pmpt", "calculate_target_return_from_goals", "validate_target_return",
        "_get_cached_frontier", "_store_frontier_results", "_optimize_max_sharpe_fallback",
        "_optimize_min_volatility_fallback", "_optimize_for_target_return", "_optimize_max_sharpe_pypfopt",
        "_optimize_risk_parity",
    ]
    for metodo in metodos:
        assert hasattr(Servicio, metodo), metodo


def test_funciones_y_presets_que_growsave_importa():
    from ypyquant.optimization.black_litterman import BlackLittermanQuantLib, crear_views_desde_dict  # noqa: F401
    from ypyquant.optimization.optimizador_portafolio import crear_modelo_personalizado, modelos_disponibles
    from ypyquant.optimization.presets import (
        PRESETS_REGISTRADOS,
        listar_presets,
        obtener_detalles_preset,
        obtener_preset,
    )

    assert callable(crear_modelo_personalizado) and callable(modelos_disponibles)
    assert callable(listar_presets) and callable(obtener_detalles_preset)
    for nombre in ("markowitz", "risk_parity", "tfm_bayes_stein", "tfm_michaud", "tfm_bl_prior"):
        assert nombre in PRESETS_REGISTRADOS
        assert obtener_preset(nombre) is not None
