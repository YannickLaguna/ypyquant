"""
ordenes.py — Decisión pura de orden a partir de una señal (sin Nautilus ni red).

Responsabilidad:
    Convertir una señal en [-1, +1] en una acción (comprar N, vender todo o nada)
    según el umbral, el tamaño máximo de posición y el equity. Es la lógica que en
    GrowSave vivía dentro de `_evaluate_order` de run.py, aislada para poder testearla.

Se relaciona con:
    - pipelines/paper_trading/alpaca_vivo.py: la usa antes de enviar la orden a Alpaca.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionOrden:
    """Resultado de decidir: `lado` es "compra", "venta" o None (sin acción)."""
    lado: str | None
    cantidad: int = 0
    cantidad_objetivo: int = 0


def decidir_orden(
    senal: float,
    umbral: float,
    equity: float,
    max_pos_pct: float,
    precio: float,
    cantidad_actual: int,
) -> DecisionOrden:
    """
    Decide la orden para un ticker.

    - Compra si senal > umbral y la cantidad objetivo (equity * max_pos_pct * |senal| / precio)
      supera la actual: compra la diferencia (mínimo 1 acción).
    - Vende toda la posición si senal < -umbral y hay posición larga.
    - En otro caso, no hace nada. Precio no positivo => sin acción.
    """
    if precio <= 0:
        return DecisionOrden(None)

    cantidad_objetivo = int(equity * max_pos_pct * abs(senal) / precio)

    if senal > umbral and cantidad_objetivo > cantidad_actual:
        a_comprar = cantidad_objetivo - cantidad_actual
        if a_comprar >= 1:
            return DecisionOrden("compra", a_comprar, cantidad_objetivo)
    elif senal < -umbral and cantidad_actual > 0:
        return DecisionOrden("venta", cantidad_actual, cantidad_objetivo)

    return DecisionOrden(None, 0, cantidad_objetivo)
