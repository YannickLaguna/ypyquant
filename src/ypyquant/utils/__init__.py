"""
Utilidades transversales de ypyquant (sin dependencias de aplicación).

- serializacion: tipos NumPy → nativos (JSON).
- divisas: contrato ServicioCambio, conversión de curvas de equity y símbolos de divisa.

Lo usan portfolio/backtest_engine y optimization/optimizador_portafolio.
"""

from ypyquant.utils.divisas import (
    SIMBOLOS_DIVISA,
    ServicioCambio,
    convertir_equity_a_divisa,
    simbolo_divisa,
)
from ypyquant.utils.serializacion import convert_numpy_to_native, convertir_numpy_a_nativo

__all__ = [
    "SIMBOLOS_DIVISA",
    "ServicioCambio",
    "convert_numpy_to_native",
    "convertir_equity_a_divisa",
    "convertir_numpy_a_nativo",
    "simbolo_divisa",
]
