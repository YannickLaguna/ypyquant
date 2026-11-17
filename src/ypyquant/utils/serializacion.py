"""
Serialización: convierte tipos NumPy a tipos nativos de Python (listos para JSON).

Responsabilidad:
- `convertir_numpy_a_nativo`: recorre dicts/listas/tuplas y convierte np.bool_, enteros,
  flotantes, cadenas y arrays de NumPy a bool/int/float/str/list nativos.

Se relaciona con:
- ypyquant.optimization.optimizador_portafolio (resultados de optimización serializables).
- ypyquant.portfolio.backtest_engine (PortfolioBacktestService._convert_numpy_types).
- GrowSave `app/utils/serialization.convert_numpy_to_native`: misma función; esta versión
  es compatible con NumPy 2 (no usa los alias retirados np.float_ / np.unicode_).
"""

from typing import Any

import numpy as np


def convertir_numpy_a_nativo(objeto: Any) -> Any:
    """
    Convierte recursivamente tipos NumPy a tipos nativos de Python.

    Args:
        objeto: valor arbitrario (dict, list, tuple, escalar NumPy, ndarray...).

    Returns:
        El mismo valor con los tipos NumPy sustituidos por nativos; el resto se devuelve igual.
    """
    if isinstance(objeto, dict):
        return {convertir_numpy_a_nativo(clave): convertir_numpy_a_nativo(valor) for clave, valor in objeto.items()}
    if isinstance(objeto, list):
        return [convertir_numpy_a_nativo(elemento) for elemento in objeto]
    if isinstance(objeto, tuple):
        return tuple(convertir_numpy_a_nativo(elemento) for elemento in objeto)
    if isinstance(objeto, np.bool_):
        return bool(objeto)
    if isinstance(objeto, np.integer):
        return int(objeto)
    if isinstance(objeto, np.floating):
        return float(objeto)
    if isinstance(objeto, np.ndarray):
        return objeto.tolist()
    if isinstance(objeto, np.str_):
        return str(objeto)
    return objeto


# Alias con el nombre que usa GrowSave
convert_numpy_to_native = convertir_numpy_a_nativo
