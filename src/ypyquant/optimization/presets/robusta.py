"""
Preset de optimización robusta con incertidumbre elipsoidal en µ (v0.5.0).

Responsabilidad: registrar `robusta_elipsoide`, un modelo listo para usar que combina µ muestral,
Σ Ledoit-Wolf y el optimizador `RobustoElipsoideOptimizer` (Goldfarb e Iyengar, 2003; Ceria y
Stubbs, 2006): maximiza µ̂'w - κ‖Θ^{1/2}w‖ - (λ/2)w'Σw con Θ = Σ/T (T = longitud de la ventana
de estimación, la que reciba `optimize`), κ = sqrt(χ²_N(90 %)) y λ = 2,5, long-only.

Convención: datos diarios anualizados con 252 (como el resto de presets). `risk_free_rate` solo
se usa para calcular el Sharpe de las métricas. Para otro κ, aversión, objetivo de volatilidad o
incertidumbre de caja construye el optimizador a mano o usa `crear_modelo_personalizado` con
`optimizer_type="robusto_elipsoide"` / `"robusto_caja"`.

Se relaciona con: presets/__init__.py (registro), portfolio_models.py (clases Strategy) y
estimacion_robusta/elipsoide.py (lógica).
"""

from ypyquant.optimization.portfolio_models import (
    HistoricalReturnsModel,
    LedoitWolfShrinkageModel,
    PortfolioOptimizationModel,
    RobustoElipsoideOptimizer,
)
from ypyquant.optimization.presets import registrar_preset


@registrar_preset('robusta_elipsoide')
def crear_robusta_elipsoide(risk_free_rate: float = 0.02) -> PortfolioOptimizationModel:
    """µ muestral + Σ Ledoit-Wolf + media-varianza robusta con elipsoide de incertidumbre (confianza 90 %)."""
    return PortfolioOptimizationModel(
        name="Robusta elipsoide",
        returns_model=HistoricalReturnsModel(method="mean", frequency=252),
        risk_model=LedoitWolfShrinkageModel(frequency=252),
        optimizer=RobustoElipsoideOptimizer(
            confianza=0.9, tipo_theta="sigma", aversion=2.5, frequency=252, risk_free_rate=risk_free_rate
        ),
        description=(
            "Optimización robusta con incertidumbre elipsoidal en µ (Goldfarb-Iyengar 2003): "
            "µ muestral, Σ Ledoit-Wolf, Θ=Σ/T y κ=sqrt(χ²_N(90 %))"
        ),
    )
