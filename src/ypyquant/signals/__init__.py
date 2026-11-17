"""
ypyquant.signals — módulo de generación de señales.
====================================================

API pública::

    from ypyquant.signals import (
        BaseSignalGenerator,
        SignalFrame,
        TechnicalSignalGenerator,
        MLSignalGenerator,
        RegimeSignalGenerator,
        FactorSignalGenerator,
        SignalCombiner,
    )

SignalFrame (alias de pd.DataFrame):
    Index   : DatetimeIndex (fechas de trading)
    Columns : ticker symbols
    Values  : float ∈ [-1.0, +1.0]
               -1 = short/underweight | 0 = neutral | +1 = long/overweight
"""

from ypyquant.signals.base import BaseSignalGenerator, SignalFrame
from ypyquant.signals.combiner import SignalCombiner
from ypyquant.signals.factor_signal import FactorSignalGenerator
from ypyquant.signals.ml_signal import MLSignalGenerator
from ypyquant.signals.regime_signal import RegimeSignalGenerator
from ypyquant.signals.technical import TechnicalSignalGenerator

__all__ = [
    "BaseSignalGenerator",
    "FactorSignalGenerator",
    "MLSignalGenerator",
    "RegimeSignalGenerator",
    "SignalCombiner",
    "SignalFrame",
    "TechnicalSignalGenerator",
]
