"""
Textos de aviso de la optimización robusta (únicos, para que todos los caminos digan lo mismo).

Responsabilidad: construir los mensajes legibles que acompañan a un resultado cuando el optimizador no
cumple lo pedido. Hoy solo uno: la volatilidad objetivo inalcanzable (σ* por debajo de la mínima varianza
posible con esos activos, long-only y sin liquidez), caso en que se devuelve la cartera de mínima varianza.

Se relaciona con:
- max_sharpe_qp.optimizar_volatilidad_objetivo y elipsoide (devuelven el estado `gmv_volatilidad_inalcanzable`).
- portfolio_models/optimizadores.py (metadatos `avisos` de los modelos), optimizador_portafolio.py (clave `aviso`
  de la fachada) y estrategias/incluidas.py (aviso agregado del experimento).
"""

from __future__ import annotations


def porcentaje_es(valor: float) -> str:
    """Fracción como porcentaje con un decimal y coma decimal española: 0.0615 -> '6,2 %' (sin locale del sistema)."""
    return f"{valor * 100:.1f}".replace(".", ",") + " %"


def aviso_volatilidad_inalcanzable(volatilidad_objetivo: float, volatilidad_minima: float) -> str:
    """
    Aviso de una sola cartera: «σ objetivo X % < σ mínima alcanzable Y %: se usa la cartera de mínima varianza».

    `volatilidad_minima` es la σ de la cartera de mínima varianza (la que se devuelve). Para bajar más hay que
    añadir liquidez (mezclar con efectivo), que el optimizador long-only totalmente invertido no hace.
    """
    return (f"σ objetivo {porcentaje_es(volatilidad_objetivo)} < σ mínima alcanzable {porcentaje_es(volatilidad_minima)}: "
            "se usa la cartera de mínima varianza")
