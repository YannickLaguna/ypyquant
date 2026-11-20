"""
ordenes.py — Decisión pura de orden a partir de una señal (sin Nautilus ni red).

Responsabilidad:
    Convertir una señal en [-1, +1] en una acción (comprar N, vender todo o nada)
    según el umbral, el tamaño máximo de posición y el equity. Es la lógica que en
    GrowSave vivía dentro de `_evaluate_order` de run.py, aislada para poder testearla.

    También `decidir_ordenes_rebalanceo`: de los pesos objetivo de una `Estrategia` del registro a la
    lista de órdenes (ventas primero) para rebalancear la cuenta, sin tocar los tickers que ya tienen una
    orden abierta (`tickers_bloqueados`).

Se relaciona con:
    - pipelines/paper_trading/alpaca_vivo.py: la usa antes de enviar la orden a Alpaca.
    - ypyquant.estrategias: de ahí salen los pesos objetivo en el modo registro.
"""

from collections.abc import Collection
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


def decidir_ordenes_rebalanceo(
    pesos: dict[str, float],
    equity: float,
    precios: dict[str, float],
    cantidades_actuales: dict[str, int],
    umbral_peso: float = 0.01,
    tickers_bloqueados: Collection[str] = (),
) -> list[tuple[str, DecisionOrden]]:
    """
    Órdenes para llevar la cuenta a los pesos objetivo (enteros de acciones, long-only).

    Para cada ticker en `pesos` o en cartera: cantidad objetivo = floor(equity * peso / precio).
    Se ignoran las diferencias inferiores a `umbral_peso` del equity (evita órdenes diminutas) salvo
    para cerrar una posición cuyo peso objetivo es 0. Las ventas van antes que las compras para
    liberar efectivo. Sin precio válido para un ticker, no se opera sobre él.

    `tickers_bloqueados`: tickers con una orden ya abierta en el broker (aún sin ejecutar). No se emite
    orden nueva sobre ellos: la cantidad actual todavía no refleja la orden en vuelo y enviar otra
    duplicaría la operación (o chocaría con un lado contrario, que Alpaca rechaza como wash trade).
    """
    ventas: list[tuple[str, DecisionOrden]] = []
    compras: list[tuple[str, DecisionOrden]] = []
    bloqueados = set(tickers_bloqueados)
    for ticker in sorted(set(pesos) | set(cantidades_actuales)):
        if ticker in bloqueados:
            continue
        precio = precios.get(ticker, 0.0)
        if precio <= 0:
            continue
        actual = int(cantidades_actuales.get(ticker, 0))
        objetivo = int(equity * max(pesos.get(ticker, 0.0), 0.0) / precio)
        diferencia = objetivo - actual
        cierra_posicion = objetivo == 0 and actual > 0
        if diferencia == 0 or (abs(diferencia) * precio < umbral_peso * equity and not cierra_posicion):
            continue
        if diferencia > 0:
            compras.append((ticker, DecisionOrden("compra", diferencia, objetivo)))
        else:
            ventas.append((ticker, DecisionOrden("venta", -diferencia, objetivo)))
    return ventas + compras
