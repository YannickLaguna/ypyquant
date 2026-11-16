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

from ypyquant.optimization.black_litterman import BlackLittermanQuantLib, View, crear_views_desde_dict
from ypyquant.portfolio.monte_carlo import MonteCarloRiesgo
from ypyquant.portfolio.bond_pricing import BondPricer
from ypyquant.data.risk_free_rates import TasaLibreRiesgoDinamica
from ypyquant.data.yield_curves import CurvaRendimientoTesoro
from ypyquant.data.quantlib_utils import (
    tiene_quantlib,
    convertir_fecha_a_ql,
    convertir_ql_a_fecha,
    crear_calendario_usa,
    obtener_day_count_convention,
    configurar_fecha_evaluacion,
    obtener_fecha_evaluacion,
    crear_quote_handle,
    obtener_periodo,
)

__all__ = [
    'BlackLittermanQuantLib', 'View', 'crear_views_desde_dict',
    'MonteCarloRiesgo',
    'BondPricer',
    'TasaLibreRiesgoDinamica',
    'CurvaRendimientoTesoro',
    'tiene_quantlib', 'convertir_fecha_a_ql', 'convertir_ql_a_fecha',
    'crear_calendario_usa', 'obtener_day_count_convention',
    'configurar_fecha_evaluacion', 'obtener_fecha_evaluacion',
    'crear_quote_handle', 'obtener_periodo',
]
