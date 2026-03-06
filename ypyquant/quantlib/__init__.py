"""
[DEPRECATED] quant.quantlib ha sido reorganizado en subpaquetes contextuales.

Este módulo reexporta desde las nuevas ubicaciones por compatibilidad con código
que ya importaba desde quant.quantlib. Los archivos originales han sido eliminados.

Nuevas ubicaciones:
  - BlackLittermanQuantLib, View  → quant.optimization.black_litterman
  - MonteCarloRiesgo              → quant.portfolio.monte_carlo
  - BondPricer                    → quant.portfolio.bond_pricing
  - TasaLibreRiesgoDinamica       → quant.data.risk_free_rates
  - CurvaRendimientoTesoro        → quant.data.yield_curves
  - tiene_quantlib, convertir_*   → quant.data.quantlib_utils
"""

import warnings

warnings.warn(
    "quant.quantlib está deprecated. "
    "Importar directamente desde quant.optimization, quant.portfolio o quant.data.",
    DeprecationWarning,
    stacklevel=2,
)

from quant.optimization.black_litterman import BlackLittermanQuantLib, View, crear_views_desde_dict
from quant.portfolio.monte_carlo import MonteCarloRiesgo
from quant.portfolio.bond_pricing import BondPricer
from quant.data.risk_free_rates import TasaLibreRiesgoDinamica
from quant.data.yield_curves import CurvaRendimientoTesoro
from quant.data.quantlib_utils import (
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
