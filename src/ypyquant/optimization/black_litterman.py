"""
Modelo Black-Litterman: combina prior de mercado con views del inversor.

Responsabilidad: estimar retornos esperados posteriores (μ_BL) mezclando el
prior de equilibrio de mercado con views subjetivas del gestor.

Flujo principal (BlackLittermanQuantLib.calcular_posterior):
  Prior Π = δ·Σ·w_mkt  →  combinar con views (P, Q, Ω)  →  posterior μ_BL.

Fórmula: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹Π + P'Ω⁻¹Q]
Las inversiones se resuelven con np.linalg.solve (más estable que inv).

Relacionado con:
  - optimization/portfolio_models.py  (BlackLittermanReturnsModel lo usa)
  - optimization/walk_forward.py      (estima μ_BL en Etapa 2B)

Ref: Black & Litterman (1992). Global Portfolio Optimization.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class View:
    """
    Representa una view del inversor sobre retornos esperados.

    Types:
        - 'absolute': View sobre retorno absoluto de un activo
          Ejemplo: "AAPL retornará 15% anual"

        - 'relative': View sobre diferencia de retornos entre activos
          Ejemplo: "AAPL superará a MSFT por 5%"
    """
    tipo: Literal['absolute', 'relative']
    activos: list[str]  # Para absolute: [ticker], para relative: [ticker1, ticker2]
    valor_esperado: float  # Retorno esperado o diferencia esperada
    confianza: float = 0.5  # Confianza en la view (0-1)

    def __post_init__(self):
        """Validar view."""
        if self.tipo == 'absolute' and len(self.activos) != 1:
            raise ValueError("View absoluta debe tener exactamente 1 activo")
        if self.tipo == 'relative' and len(self.activos) != 2:
            raise ValueError("View relativa debe tener exactamente 2 activos")
        if not 0 <= self.confianza <= 1:
            raise ValueError("Confianza debe estar entre 0 y 1")


class BlackLittermanQuantLib:
    """
    Implementación completa del modelo Black-Litterman.

    I: Σ (covarianza), pesos de mercado, lista de View, τ, δ.
    T: Construye P y Q desde las views; calcula Ω (Idzorek/proporcional/igual);
       resuelve el sistema lineal con np.linalg.solve; valida positividad de Ω.
    O: Series μ_BL — retornos esperados posteriores por ticker.
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.025,
        metodo_omega: Literal['idzorek', 'proportional', 'equal'] = 'idzorek'
    ):
        """
        Inicializa el modelo Black-Litterman.

        Args:
            risk_aversion: Coeficiente de aversión al riesgo del mercado (δ)
                          Típicamente entre 2.0 y 4.0
            tau: Parámetro de incertidumbre del prior (τ)
                 Típicamente entre 0.01 y 0.05
                 Valores menores = mayor confianza en el equilibrio
            metodo_omega: Método para calcular matriz Omega de incertidumbre
                         - 'idzorek': Método de Idzorek (recomendado)
                         - 'proportional': Proporcional a covarianza
                         - 'equal': Varianza igual para todas las views
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.metodo_omega = metodo_omega

    def calcular_retornos_equilibrio(
        self,
        cov_matrix: pd.DataFrame,
        market_weights: pd.Series
    ) -> pd.Series:
        """
        Prior de mercado: Π = δ·Σ·w_mkt.

        I: Σ (DataFrame), pesos de mercado (Series, se normalizan a 1).
        T: Producto matricial δ·Σ·w.
        O: Series con retornos de equilibrio implícitos del mercado.
        """
        # Asegurar que activos estén alineados
        activos_comunes = cov_matrix.index.intersection(market_weights.index)
        cov_matrix = cov_matrix.loc[activos_comunes, activos_comunes]
        market_weights = market_weights.loc[activos_comunes]

        # Normalizar pesos
        market_weights = market_weights / market_weights.sum()

        # Calcular prior: Π = δ * Σ * w
        prior = self.risk_aversion * cov_matrix.dot(market_weights)

        return prior

    def construir_matrices_views(
        self,
        views: list[View],
        activos: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construye matrices P (pick) y Q (valores esperados) desde views.

        Args:
            views: Lista de views del inversor
            activos: Lista de todos los activos

        Returns:
            Tuple (P, Q) donde:
            - P: Pick matrix (K x N) que selecciona activos en cada view
            - Q: Vector (K,) con valores esperados de cada view
        """
        K = len(views)  # Número de views
        N = len(activos)  # Número de activos

        P = np.zeros((K, N))
        Q = np.zeros(K)

        activo_to_idx = {activo: i for i, activo in enumerate(activos)}

        for i, view in enumerate(views):
            Q[i] = view.valor_esperado

            if view.tipo == 'absolute':
                # View absoluta: P[i, j] = 1 si el activo j está en la view
                activo = view.activos[0]
                if activo in activo_to_idx:
                    P[i, activo_to_idx[activo]] = 1.0

            elif view.tipo == 'relative':
                # View relativa: P[i, j1] = 1, P[i, j2] = -1
                activo1, activo2 = view.activos
                if activo1 in activo_to_idx and activo2 in activo_to_idx:
                    P[i, activo_to_idx[activo1]] = 1.0
                    P[i, activo_to_idx[activo2]] = -1.0

        return P, Q

    def calcular_omega(
        self,
        views: list[View],
        P: np.ndarray,
        cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """
        Matriz diagonal Ω de incertidumbre de views (K×K).

        I: views (con confianza 0-1), P (pick matrix K×N), Σ.
        T: Según metodo_omega — Idzorek: view_var/confianza;
           proporcional: τ·P·Σ·P'; igual: τ·I.
        O: np.ndarray diagonal (K×K) — valores menores = mayor confianza.
        """
        K = len(views)
        Sigma = cov_matrix.values

        if self.metodo_omega == 'idzorek':
            # Método de Idzorek: Ω = diag(P * τΣ * P') / confianza
            omega_diag = np.zeros(K)
            for i, view in enumerate(views):
                # Varianza de la view
                view_var = P[i, :] @ (self.tau * Sigma) @ P[i, :].T
                # Ajustar por confianza (mayor confianza = menor varianza)
                omega_diag[i] = view_var / max(view.confianza, 0.01)

        elif self.metodo_omega == 'proportional':
            # Proporcional a covarianza: Ω = τ * P * Σ * P'
            omega_matrix = self.tau * P @ Sigma @ P.T
            omega_diag = np.diag(omega_matrix)

        elif self.metodo_omega == 'equal':
            # Varianza igual para todas las views
            omega_diag = np.ones(K) * self.tau

        else:
            raise ValueError(f"Método omega desconocido: {self.metodo_omega}")

        return np.diag(omega_diag)

    def calcular_posterior(
        self,
        cov_matrix: pd.DataFrame,
        market_weights: pd.Series,
        views: list[View] | None = None,
        prior: pd.Series | None = None
    ) -> pd.Series:
        """
        Posterior Black-Litterman: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹·[(τΣ)⁻¹Π + P'Ω⁻¹Q].

        I: Σ, pesos de mercado, lista de View (opcional), prior personalizado (opcional).
        T: Construye P/Q/Ω → valida positividad de Ω → resuelve sistema lineal con solve().
           Sin views retorna el prior directamente.
        O: Series μ_BL con retornos esperados posteriores por ticker.
        """
        # Calcular prior si no se proporciona
        if prior is None:
            prior = self.calcular_retornos_equilibrio(cov_matrix, market_weights)

        # Si no hay views, retornar prior
        if not views or len(views) == 0:
            return prior

        # Preparar matrices
        activos = list(cov_matrix.index)
        Sigma = cov_matrix.values
        Pi = prior.values

        # Construir P y Q desde views
        P, Q = self.construir_matrices_views(views, activos)

        # Calcular Omega
        Omega = self.calcular_omega(views, P, cov_matrix)

        # Validar positividad definida de Omega
        eigvals_omega = np.linalg.eigvalsh(Omega)
        if not np.all(eigvals_omega > 0):
            logger.warning("[BL] Omega no es positiva definida (autovalores mínimos: "
                           f"{eigvals_omega.min():.2e}). Resultados pueden ser inestables.")

        # Fórmula Black-Litterman
        # μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹Π + P'Ω⁻¹Q]
        # Usamos np.linalg.solve en lugar de inv() explícita para mayor estabilidad numérica.

        tau_Sigma = self.tau * Sigma

        # (τΣ)⁻¹ Π  →  solve(τΣ, Π)
        tau_Sigma_inv_Pi = np.linalg.solve(tau_Sigma, Pi)

        # Ω⁻¹ Q  →  solve(Ω, Q)
        Omega_inv_Q = np.linalg.solve(Omega, Q)

        # Ω⁻¹ aplicado a cada columna de P'  →  solve(Ω, P)
        Omega_inv_P = np.linalg.solve(Omega, P)  # shape (K, N)

        # Parte 1: M_inv = (τΣ)⁻¹ + P'Ω⁻¹P  →  solve(τΣ, I) + P'·Omega_inv_P
        tau_Sigma_inv = np.linalg.solve(tau_Sigma, np.eye(len(Sigma)))
        M_inv = tau_Sigma_inv + P.T @ Omega_inv_P

        # Parte 2: b = (τΣ)⁻¹Π + P'Ω⁻¹Q
        b = tau_Sigma_inv_Pi + P.T @ Omega_inv_Q

        # Posterior: M_inv · μ_BL = b  →  solve(M_inv, b)
        mu_BL = np.linalg.solve(M_inv, b)

        return pd.Series(mu_BL, index=activos, name='expected_returns_bl')

    def calcular_posterior_cov(
        self,
        cov_matrix: pd.DataFrame,
        views: list[View] | None = None
    ) -> pd.DataFrame:
        """
        Calcula matriz de covarianza posterior.

        La covarianza posterior incorpora la reducción de incertidumbre
        por las views del inversor.

        Σ_BL = Σ + [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹

        Args:
            cov_matrix: Matriz de covarianza prior
            views: Lista de views (opcional)

        Returns:
            Matriz de covarianza posterior
        """
        if not views or len(views) == 0:
            return cov_matrix

        activos = list(cov_matrix.index)
        Sigma = cov_matrix.values

        # Construir P y Omega
        P, _ = self.construir_matrices_views(views, activos)
        Omega = self.calcular_omega(views, P, cov_matrix)

        # Calcular componentes usando solve() en lugar de inv() explícita
        tau_Sigma = self.tau * Sigma
        tau_Sigma_inv = np.linalg.solve(tau_Sigma, np.eye(len(Sigma)))
        Omega_inv_P = np.linalg.solve(Omega, P)  # shape (K, N)

        # Σ_BL = Σ + M donde M = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
        M_inv = tau_Sigma_inv + P.T @ Omega_inv_P
        M = np.linalg.solve(M_inv, np.eye(len(Sigma)))

        Sigma_BL = Sigma + M

        return pd.DataFrame(Sigma_BL, index=activos, columns=activos)

    def analizar_impacto_views(
        self,
        prior: pd.Series,
        posterior: pd.Series,
        views: list[View]
    ) -> pd.DataFrame:
        """
        Analiza el impacto de las views en los retornos esperados.

        Args:
            prior: Retornos de equilibrio (antes de views)
            posterior: Retornos posteriores (después de views)
            views: Views aplicadas

        Returns:
            DataFrame con análisis de cambios
        """
        cambios = pd.DataFrame({
            'prior': prior,
            'posterior': posterior,
            'cambio_absoluto': posterior - prior,
            'cambio_relativo': (posterior - prior) / prior.abs()
        })

        # Marcar qué activos tienen views
        activos_con_views = set()
        for view in views:
            activos_con_views.update(view.activos)

        cambios['tiene_view'] = cambios.index.isin(activos_con_views)

        # Ordenar por cambio absoluto
        cambios = cambios.sort_values('cambio_absoluto', ascending=False, key=abs)

        return cambios


def crear_views_desde_dict(views_dict: list[dict]) -> list[View]:
    """
    Crea objetos View desde diccionarios.

    Helper function para facilitar creación de views desde API/UI.

    Args:
        views_dict: Lista de diccionarios con estructura:
            [
                {
                    'tipo': 'absolute',
                    'activos': ['AAPL'],
                    'valor_esperado': 0.15,
                    'confianza': 0.7
                },
                {
                    'tipo': 'relative',
                    'activos': ['MSFT', 'GOOGL'],
                    'valor_esperado': 0.05,
                    'confianza': 0.5
                }
            ]

    Returns:
        Lista de objetos View

    Example:
        >>> views_dict = [
        ...     {'tipo': 'absolute', 'activos': ['AAPL'], 'valor_esperado': 0.15, 'confianza': 0.7}
        ... ]
        >>> views = crear_views_desde_dict(views_dict)
    """
    views = []
    for v in views_dict:
        view = View(
            tipo=v['tipo'],
            activos=v['activos'],
            valor_esperado=v['valor_esperado'],
            confianza=v.get('confianza', 0.5)
        )
        views.append(view)

    return views
