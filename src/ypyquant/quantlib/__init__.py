"""
[DEPRECATED] ypyquant.quantlib ha sido reorganizado en subpaquetes contextuales.

Este módulo reexporta desde las nuevas ubicaciones por compatibilidad con código
que ya importaba desde ypyquant.quantlib. Los archivos originales han sido eliminados.

Nuevas ubicaciones:
  - BlackLittermanQuantLib, View  → ypyquant.optimization.black_litterman
  - MonteCarloRiesgo              → ypyquant.portfolio.monte_carlo
  - BondPricer                    → ypyquant.portfolio.bond_pricing
  - TasaLibreRiesgoDinamica       → ypyquant.data.risk_free_rates
  - CurvaRendimientoTesoro        → ypyquant.data.yield_curves
  - tiene_quantlib, convertir_*   → ypyquant.data.quantlib_utils
"""

import warnings

warnings.warn(
    "ypyquant.quantlib está deprecated. "
    "Importar directamente desde ypyquant.optimization, ypyquant.portfolio o ypyquant.data.",
    DeprecationWarning,
    stacklevel=2,
)

from ypyquant.data.quantlib_utils import (
    configurar_fecha_evaluacion,
    convertir_fecha_a_ql,
    convertir_ql_a_fecha,
    crear_calendario_usa,
    crear_quote_handle,
    obtener_day_count_convention,
    obtener_fecha_evaluacion,
    obtener_periodo,
    tiene_quantlib,
)
from ypyquant.data.risk_free_rates import TasaLibreRiesgoDinamica
from ypyquant.data.yield_curves import CurvaRendimientoTesoro
from ypyquant.optimization.black_litterman import BlackLittermanQuantLib, View, crear_views_desde_dict
from ypyquant.portfolio.bond_pricing import BondPricer
from ypyquant.portfolio.monte_carlo import MonteCarloRiesgo

__all__ = [
    'BlackLittermanQuantLib',
    'BondPricer',
    'CurvaRendimientoTesoro',
    'MonteCarloRiesgo',
    'TasaLibreRiesgoDinamica',
    'View',
    'configurar_fecha_evaluacion',
    'convertir_fecha_a_ql',
    'convertir_ql_a_fecha',
    'crear_calendario_usa',
    'crear_quote_handle',
    'crear_views_desde_dict',
    'obtener_day_count_convention',
    'obtener_fecha_evaluacion',
    'obtener_periodo',
    'tiene_quantlib',
]
