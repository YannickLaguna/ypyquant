"""
constructor_senales.py — Construye un SignalCombiner de ypyquant desde un dict de configuración.

Responsabilidad:
    Traducir `{"generators": [{"name": "technical", "weight": 1.0, "params": {...}}]}`
    a un `SignalCombiner` con sus generadores y pesos. No depende de Nautilus ni de red.

Se relaciona con:
    - ypyquant.signals: aporta los generadores y el combinador.
    - estrategia_ypyquant.py y pipelines/paper_trading: consumen `construir_combinador`.
"""

import logging

from ypyquant.signals import (
    FactorSignalGenerator,
    MLSignalGenerator,
    RegimeSignalGenerator,
    SignalCombiner,
    TechnicalSignalGenerator,
)

logger = logging.getLogger(__name__)

# Nombre usado en la config -> clase generadora
GENERADORES = {
    "technical": TechnicalSignalGenerator,
    "factor": FactorSignalGenerator,
    "ml": MLSignalGenerator,
    "regime": RegimeSignalGenerator,
}


def construir_combinador(config_senales: dict) -> SignalCombiner:
    """
    Crea un SignalCombiner a partir de la config de señales de una estrategia.

    Los generadores desconocidos se ignoran con un aviso; si no queda ninguno
    (o la config está vacía) se usa el generador técnico con peso 1.

    Args:
        config_senales: dict con la clave opcional "generators" (lista de
            {"name", "weight", "params"}).

    Returns:
        SignalCombiner listo para llamar a `.generate(precios)`.
    """
    config_generadores = config_senales.get("generators", [{"name": "technical", "weight": 1.0}])
    generadores, pesos = [], []

    for cfg in config_generadores:
        nombre = cfg["name"]
        clase = GENERADORES.get(nombre)
        if clase is None:
            logger.warning("Generador desconocido '%s', ignorando", nombre)
            continue
        generadores.append(clase(**cfg.get("params", {})))
        pesos.append(cfg.get("weight", 1.0))

    if not generadores:
        generadores = [TechnicalSignalGenerator()]
        pesos = [1.0]

    return SignalCombiner(generators=generadores, weights=pesos)
