"""
Dimensionamiento de las entradas de una estrategia de señales (función pura).

Responsabilidad: dado un grupo de candidatos elegidos para entrar, devolver el peso (magnitud positiva) de cada uno según
la política (`Dimensionamiento`): igual, inverso a la volatilidad, proporcional a la fuerza de la señal o por riesgo por
operación. NO decide cuántas posiciones caben ni la exposición total: eso son los límites (`LimitesExposicion`) que aplica
la máquina de estados al escalar el conjunto. Con `iguales` e `inversa_volatilidad` reproduce exactamente el cálculo del
marco anterior (solo largo).

Se relaciona con: .senales_modelo (`Dimensionamiento`), .senales_maquina (la usa al abrir posiciones) y .senales (parámetros).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

from ypyquant.estrategias.senales_modelo import Dimensionamiento


def _relativos_por_media(valores: Mapping[str, float]) -> dict[str, float]:
    """Cada valor dividido por la media de los válidos (finitos y > 0); los inválidos reciben el tamaño medio (1)."""
    validos = {t: v for t, v in valores.items() if math.isfinite(v) and v > 0}
    if not validos:
        return {t: 1.0 for t in valores}
    media = sum(validos.values()) / len(validos)
    return {t: validos.get(t, media) / media for t in valores}


def tamanos_de_entrada(elegidos: list[str], fuerzas: Mapping[str, float | None], volatilidad: Callable[[str], float | None],
                       dimension: Dimensionamiento) -> dict[str, float]:
    """
    Peso de entrada de cada elegido (positivo, con el tope `peso_maximo`).

    - `iguales`: 1/max_posiciones. `inversa_volatilidad`: repartido por 1/vol (sin volatilidad medible: tamaño medio).
    - `fuerza`: proporcional a la fuerza de la señal (la ausente o no positiva recibe el tamaño medio).
    - `riesgo`: `riesgo_por_operacion / stop_perdida` (exige stop de pérdida); si no hay stop, cae a `iguales`.
    """
    base = 1.0 / dimension.max_posiciones
    relativos = {t: 1.0 for t in elegidos}
    if dimension.ponderacion == "inversa_volatilidad":
        inversas = {}
        for t in elegidos:
            vol = volatilidad(t)
            if vol is not None and math.isfinite(vol) and vol > 0:
                inversas[t] = 1.0 / vol
        if inversas:
            media = sum(inversas.values()) / len(inversas)
            relativos = {t: inversas.get(t, media) / media for t in elegidos}  # sin volatilidad medible: tamaño medio
    elif dimension.ponderacion == "fuerza":
        relativos = _relativos_por_media({t: (fuerzas.get(t) if fuerzas.get(t) is not None else math.nan) for t in elegidos})
    elif dimension.ponderacion == "riesgo" and dimension.stop_perdida:
        base = dimension.riesgo_por_operacion / dimension.stop_perdida
    return {t: min(base * relativos[t], dimension.peso_maximo) for t in elegidos}
