"""
Registro central de presets de optimización de portafolios.

Cada preset puede definirse en un archivo separado y registrarse aquí.
Esto permite mantener el código organizado y facilita agregar nuevos presets.

Estructura:
    presets/
        __init__.py          ← Registro central (este archivo)
        tradicionales.py     ← Presets clásicos (Markowitz, Risk Parity, etc.)
        factor_models.py     ← Presets basados en factores (CAPM, FF4, BL)
        ml_avanzados.py      ← Presets con Machine Learning (LSTM, PCA, SVM)
        pmpt.py              ← Presets Post-Modern Portfolio Theory
        tfm.py               ← Presets del TFM (Bayes-Stein, Michaud, BL-prior)
        robusta.py           ← Preset de incertidumbre elipsoidal en µ (robusta_elipsoide)

Uso:
    from ypyquant.optimization.presets import obtener_preset, listar_presets

    # Listar todos los presets disponibles
    presets = listar_presets()

    # Obtener un preset específico
    modelo = obtener_preset('markowitz')
"""

from collections.abc import Callable
from typing import Any, Dict

from ypyquant.optimization.portfolio_models import PortfolioOptimizationModel

# Diccionario central de presets registrados
# Formato: {nombre_preset: funcion_factory}
PRESETS_REGISTRADOS: dict[str, Callable] = {}


def registrar_preset(nombre: str):
    """
    Decorador para registrar un preset en el registro central.

    Ejemplo:
        @registrar_preset('mi_preset')
        def crear_mi_preset() -> PortfolioOptimizationModel:
            return PortfolioOptimizationModel(...)
    """
    def decorador(funcion_factory: Callable):
        PRESETS_REGISTRADOS[nombre] = funcion_factory
        return funcion_factory
    return decorador


def obtener_preset(nombre_preset: str, risk_free_rate: float = 0.02):
    """
    Obtiene un preset por nombre.

    Args:
        nombre_preset: Nombre del preset
        risk_free_rate: Tasa libre de riesgo a usar (default: 0.02)

    Returns:
        PortfolioOptimizationModel o Dict con configuración

    Raises:
        ValueError: Si el preset no existe
    """
    if nombre_preset not in PRESETS_REGISTRADOS:
        disponibles = ', '.join(PRESETS_REGISTRADOS.keys())
        raise ValueError(
            f"Preset '{nombre_preset}' no existe. "
            f"Disponibles: {disponibles}"
        )

    return PRESETS_REGISTRADOS[nombre_preset](risk_free_rate=risk_free_rate)


def listar_presets() -> dict[str, str]:
    """
    Lista todos los presets disponibles con descripciones.

    Returns:
        Dict con {nombre_preset: descripcion}
    """
    info_presets = {}

    for nombre, factory in PRESETS_REGISTRADOS.items():
        preset = factory(risk_free_rate=0.02)  # Default para listing

        # Extraer descripción según tipo de preset
        if isinstance(preset, PortfolioOptimizationModel):
            descripcion = preset.description
        else:
            descripcion = preset.get('description', 'Sin descripción')

        info_presets[nombre] = descripcion

    return info_presets


def obtener_detalles_preset(nombre_preset: str, risk_free_rate: float = 0.02) -> dict:
    """
    Obtiene detalles completos de un preset.

    Args:
        nombre_preset: Nombre del preset
        risk_free_rate: Tasa libre de riesgo a usar (default: 0.02)

    Returns:
        Dict con toda la información del preset
    """
    preset = obtener_preset(nombre_preset, risk_free_rate=risk_free_rate)

    if isinstance(preset, PortfolioOptimizationModel):
        # Preset tradicional
        return {
            'nombre': preset.name,
            'descripcion': preset.description,
            'modelo_retornos': preset.returns_model.to_dict(),
            'modelo_riesgo': preset.risk_model.to_dict(),
            'optimizador': preset.optimizer.to_dict(),
            'tipo': 'tradicional'
        }
    else:
        # Preset avanzado (dict)
        preset['tipo'] = 'avanzado'
        return preset


# ============================================================================
# IMPORTAR Y REGISTRAR TODOS LOS PRESETS
# ============================================================================

# Importar presets de cada módulo (esto los registra automáticamente)
from ypyquant.optimization.presets import factor_models, pmpt, robusta, tfm, tradicionales  # noqa: F401 - registran los presets

# Los presets ML avanzados se importan condicionalmente
try:
    from ypyquant.optimization.presets import ml_avanzados
except ImportError:
    # No hay dependencias ML instaladas, skip
    pass
