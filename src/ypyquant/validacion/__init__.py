"""
Validación estadística de estrategias: librería pura (numpy/pandas/scipy).

Responsabilidad:
- Reexportar la API pública de `estadisticos` (Sharpe, PSR, DSR, PBO-CSCV, rachas, Ulcer/Martin,
  p-valor de permutación), `permutacion` (precios permutados, método de Masters) y
  `walk_forward` (selección walk-forward sobre una matriz de retornos).

Se relaciona con:
- ypyquant.estrategias.experimentos (conexión posterior: matriz de retornos de los barridos y
  `registrar_validador`).
- ypyquant.data.matriz_precios (formato de precios que consume `permutar_precios`).
- docs/validacion.md (fórmulas, referencias e interpretación).
"""

from ypyquant.validacion.estadisticos import (
    indice_ulcer,
    p_valor_permutacion,
    pbo_cscv,
    prueba_rachas,
    ratio_martin,
    sharpe,
    sharpe_deflactado,
    sharpe_maximo_esperado,
    sharpe_probabilistico,
)
from ypyquant.validacion.permutacion import permutar_precios
from ypyquant.validacion.walk_forward import seleccion_walk_forward

__all__ = [
    "indice_ulcer",
    "p_valor_permutacion",
    "pbo_cscv",
    "permutar_precios",
    "prueba_rachas",
    "ratio_martin",
    "seleccion_walk_forward",
    "sharpe",
    "sharpe_deflactado",
    "sharpe_maximo_esperado",
    "sharpe_probabilistico",
]
