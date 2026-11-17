"""
Risk Tolerance Profile Configuration Service

Este módulo define cómo los perfiles de tolerancia al riesgo (conservative, moderate, aggressive)
afectan los parámetros de optimización de portafolios.

Cada perfil tiene configuraciones específicas para:
- Restricciones de pesos (max_weight, min_weight)
- Tipo de optimizador preferido
- Volatilidad objetivo
- Configuración de cash allocation
- Restricciones adicionales
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskProfile:
    """
    Configuración completa de un perfil de riesgo.

    Attributes:
        name: Nombre del perfil
        display_name: Nombre mostrado al usuario
        description: Descripción del perfil
        max_single_asset: Peso máximo por activo individual (0-1)
        min_single_asset: Peso mínimo por activo si se incluye (0-1)
        preferred_optimizer: Optimizador recomendado
        target_volatility_range: Rango de volatilidad objetivo (min, max)
        cash_allocation_recommended: Si se recomienda incluir cash
        diversification_level: Nivel de diversificación (low, medium, high)
        risk_aversion_coefficient: Coeficiente de aversión al riesgo para optimizadores
        constraints: Restricciones adicionales por defecto
    """
    name: str
    display_name: str
    description: str
    max_single_asset: float
    min_single_asset: float
    preferred_optimizer: str
    target_volatility_range: tuple[float, float]
    cash_allocation_recommended: bool
    diversification_level: str
    risk_aversion_coefficient: float
    constraints: dict[str, Any] = field(default_factory=dict)

    def get_optimizer_config(self, risk_free_rate: float = 0.02) -> dict[str, Any]:
        """
        Genera configuración del optimizador según el perfil.

        Args:
            risk_free_rate: Tasa libre de riesgo

        Returns:
            Dict con parámetros para el optimizador
        """
        config = {
            "risk_free_rate": risk_free_rate,
            "risk_aversion": self.risk_aversion_coefficient
        }

        # Para perfiles conservadores, agregar penalización por volatilidad
        if self.name == "conservative":
            config["volatility_penalty"] = 1.5

        return config

    def get_constraints(self, user_constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Combina constraints del perfil con constraints personalizados del usuario.

        Args:
            user_constraints: Restricciones personalizadas del usuario

        Returns:
            Dict con constraints finales
        """
        # Empezar con constraints del perfil
        final_constraints = {
            "max_weight": self.max_single_asset,
            "min_weight": self.min_single_asset,
            **self.constraints
        }

        # Sobrescribir con constraints del usuario si existen
        if user_constraints:
            final_constraints.update(user_constraints)

        return final_constraints

    def get_target_volatility(self, frontier_volatility: float) -> float | None:
        """
        Calcula volatilidad objetivo según el perfil y la frontera eficiente.

        Args:
            frontier_volatility: Volatilidad del portafolio de tangencia

        Returns:
            Volatilidad objetivo ajustada según perfil, o None si no se recomienda cash
        """
        min_vol, max_vol = self.target_volatility_range

        # Ajustar según el rango del perfil
        if self.name == "conservative":
            # Conservative: apuntar al límite inferior
            return max(min_vol, frontier_volatility * 0.7)
        elif self.name == "moderate":
            # Moderate: usar volatilidad de tangencia o ajustar ligeramente
            return min(max_vol, frontier_volatility)
        else:  # aggressive
            # Aggressive: no reducir volatilidad (o incluso aumentar con leverage)
            # Por ahora no soportamos leverage, así que retornar None
            return None


# ============================================================================
# PERFILES PREDEFINIDOS
# ============================================================================

CONSERVATIVE_PROFILE = RiskProfile(
    name="conservative",
    display_name="Conservador",
    description="Prioriza la seguridad del capital. Prefiere menor riesgo aunque los retornos sean moderados. "
                "Alta diversificación y baja concentración.",
    max_single_asset=0.20,  # Máximo 20% en un solo activo
    min_single_asset=0.02,  # Mínimo 2% si se incluye (al menos 5 activos si hay 100% invertido)
    preferred_optimizer="min_volatility",  # Minimizar volatilidad
    target_volatility_range=(0.05, 0.12),  # 5-12% anualizado
    cash_allocation_recommended=True,
    diversification_level="high",
    risk_aversion_coefficient=3.0,  # Alta aversión al riesgo
    constraints={
        "long_only": True,  # Solo posiciones largas (no short selling)
        "sector_max_weight": 0.30,  # Máximo 30% en un sector (si se implementa)
    }
)

MODERATE_PROFILE = RiskProfile(
    name="moderate",
    display_name="Moderado",
    description="Busca balance entre riesgo y retorno. Acepta cierta volatilidad para mejores rendimientos. "
                "Diversificación equilibrada.",
    max_single_asset=0.30,  # Máximo 30% en un solo activo
    min_single_asset=0.03,  # Mínimo 3% si se incluye
    preferred_optimizer="max_sharpe",  # Maximizar ratio Sharpe
    target_volatility_range=(0.10, 0.20),  # 10-20% anualizado
    cash_allocation_recommended=False,  # Cash opcional
    diversification_level="medium",
    risk_aversion_coefficient=1.5,  # Aversión moderada
    constraints={
        "long_only": True,
        "sector_max_weight": 0.40,
    }
)

AGGRESSIVE_PROFILE = RiskProfile(
    name="aggressive",
    display_name="Agresivo",
    description="Busca maximizar retornos. Puede tolerar alta volatilidad y riesgo de pérdidas temporales. "
                "Permite mayor concentración en activos de alta convicción.",
    max_single_asset=0.40,  # Máximo 40% en un solo activo
    min_single_asset=0.05,  # Mínimo 5% si se incluye (más concentrado)
    preferred_optimizer="max_sharpe",  # Maximizar Sharpe, pero tolera más riesgo
    target_volatility_range=(0.15, 0.35),  # 15-35% anualizado (tolera alta volatilidad)
    cash_allocation_recommended=False,  # Full investment, sin cash
    diversification_level="low",
    risk_aversion_coefficient=0.5,  # Baja aversión (más tolerante al riesgo)
    constraints={
        "long_only": True,
        "sector_max_weight": 0.50,  # Permite mayor concentración sectorial
    }
)


# Registro de perfiles disponibles
RISK_PROFILES: dict[str, RiskProfile] = {
    "conservative": CONSERVATIVE_PROFILE,
    "moderate": MODERATE_PROFILE,
    "aggressive": AGGRESSIVE_PROFILE
}


# ============================================================================
# SERVICIO DE PERFILES DE RIESGO
# ============================================================================

class RiskProfileService:
    """
    Servicio para gestionar perfiles de riesgo y aplicarlos a optimizaciones.
    """

    @staticmethod
    def get_profile(risk_tolerance: str) -> RiskProfile:
        """
        Obtiene el perfil de riesgo por nombre.

        Args:
            risk_tolerance: Nombre del perfil ('conservative', 'moderate', 'aggressive')

        Returns:
            RiskProfile correspondiente

        Raises:
            ValueError: Si el perfil no existe
        """
        profile = RISK_PROFILES.get(risk_tolerance.lower())
        if not profile:
            raise ValueError(
                f"Unknown risk tolerance: {risk_tolerance}. "
                f"Valid options: {', '.join(RISK_PROFILES.keys())}"
            )
        return profile

    @staticmethod
    def get_all_profiles() -> dict[str, RiskProfile]:
        """
        Obtiene todos los perfiles disponibles.

        Returns:
            Dict con todos los perfiles indexados por nombre
        """
        return RISK_PROFILES

    @staticmethod
    def get_profile_summary(risk_tolerance: str) -> dict[str, Any]:
        """
        Obtiene un resumen del perfil en formato JSON-serializable.

        Args:
            risk_tolerance: Nombre del perfil

        Returns:
            Dict con información del perfil
        """
        profile = RiskProfileService.get_profile(risk_tolerance)

        return {
            "name": profile.name,
            "display_name": profile.display_name,
            "description": profile.description,
            "parameters": {
                "max_single_asset": f"{profile.max_single_asset * 100:.0f}%",
                "min_single_asset": f"{profile.min_single_asset * 100:.0f}%",
                "preferred_optimizer": profile.preferred_optimizer,
                "target_volatility_range": f"{profile.target_volatility_range[0] * 100:.0f}%-{profile.target_volatility_range[1] * 100:.0f}%",
                "diversification_level": profile.diversification_level,
                "cash_recommended": profile.cash_allocation_recommended
            },
            "constraints": profile.constraints
        }

    @staticmethod
    def apply_profile_to_optimization(
        risk_tolerance: str,
        base_config: dict[str, Any],
        user_constraints: dict[str, Any] | None = None,
        risk_free_rate: float = 0.02
    ) -> dict[str, Any]:
        """
        Aplica configuración de perfil de riesgo a parámetros de optimización.

        Args:
            risk_tolerance: Perfil de tolerancia al riesgo
            base_config: Configuración base de optimización
            user_constraints: Restricciones adicionales del usuario
            risk_free_rate: Tasa libre de riesgo

        Returns:
            Dict con configuración ajustada según perfil
        """
        profile = RiskProfileService.get_profile(risk_tolerance)

        # Combinar configuración
        optimized_config = dict(base_config)

        # Aplicar optimizador preferido si no se especificó uno
        if "optimizer_type" not in optimized_config or not optimized_config["optimizer_type"]:
            optimized_config["optimizer_type"] = profile.preferred_optimizer

        # Aplicar constraints del perfil
        optimized_config["constraints"] = profile.get_constraints(user_constraints)

        # Configurar optimizer
        if "optimizer_config" not in optimized_config:
            optimized_config["optimizer_config"] = {}

        optimized_config["optimizer_config"].update(
            profile.get_optimizer_config(risk_free_rate)
        )

        # Información del perfil aplicado
        optimized_config["applied_risk_profile"] = {
            "name": profile.name,
            "display_name": profile.display_name,
            "diversification_level": profile.diversification_level
        }

        return optimized_config

    @staticmethod
    def recommend_cash_allocation(
        risk_tolerance: str,
        portfolio_volatility: float,
        risk_free_rate: float = 0.02
    ) -> dict[str, Any]:
        """
        Recomienda si incluir cash y qué volatilidad objetivo según perfil.

        Args:
            risk_tolerance: Perfil de riesgo
            portfolio_volatility: Volatilidad del portafolio optimizado (tangencia)
            risk_free_rate: Tasa libre de riesgo

        Returns:
            Dict con recomendaciones de cash allocation
        """
        profile = RiskProfileService.get_profile(risk_tolerance)

        recommendation = {
            "should_include_cash": profile.cash_allocation_recommended,
            "reason": ""
        }

        if profile.cash_allocation_recommended:
            target_vol = profile.get_target_volatility(portfolio_volatility)

            if target_vol and target_vol < portfolio_volatility:
                # Necesita cash para reducir volatilidad
                cash_weight = 1 - (target_vol / portfolio_volatility)

                recommendation.update({
                    "target_volatility": target_vol,
                    "estimated_cash_weight": cash_weight,
                    "reason": f"El perfil {profile.display_name} recomienda volatilidad de {target_vol*100:.1f}%, "
                              f"lo que requiere {cash_weight*100:.1f}% en efectivo/bonos."
                })
            else:
                recommendation["reason"] = f"La volatilidad del portafolio ({portfolio_volatility*100:.1f}%) ya está dentro del rango objetivo."
        else:
            recommendation["reason"] = f"El perfil {profile.display_name} no recomienda cash allocation (full investment)."

        return recommendation


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_profile_for_user(user_risk_tolerance: str | None = None) -> RiskProfile:
    """
    Obtiene el perfil de riesgo de un usuario (con fallback a moderado).

    Args:
        user_risk_tolerance: Tolerancia al riesgo del usuario

    Returns:
        RiskProfile correspondiente
    """
    if not user_risk_tolerance:
        return MODERATE_PROFILE

    return RiskProfileService.get_profile(user_risk_tolerance)


def validate_risk_tolerance(risk_tolerance: str) -> bool:
    """
    Valida que un string de tolerancia al riesgo sea válido.

    Args:
        risk_tolerance: String a validar

    Returns:
        True si es válido, False en caso contrario
    """
    return risk_tolerance.lower() in RISK_PROFILES
