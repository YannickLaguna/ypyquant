"""
Modelo de riesgo basado en PCA (Principal Component Analysis).
Reduce ruido en matrices de covarianza mediante reducción de dimensionalidad.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class PCARiskService:
    """
    Servicio para cálculo de matrices de covarianza usando PCA.

    PCA elimina ruido al:
    1. Identificar componentes principales de varianza
    2. Eliminar componentes de bajo impacto (ruido)
    3. Reconstruir matriz de covarianza "limpia"
    """

    def __init__(self, n_components: float = 0.95):
        """
        Args:
            n_components: Varianza a explicar (0.0-1.0) o número de componentes (int)
                - 0.95 = explicar 95% de la varianza
                - 10 = usar exactamente 10 componentes
        """
        self.n_components = n_components
        self.pca = None
        self.explained_variance_ratio = None
        self.components = None

    def fit_transform(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica PCA a los retornos.

        Args:
            returns_df: DataFrame con retornos diarios (índice=fechas, columnas=tickers)

        Returns:
            DataFrame con retornos transformados a componentes principales
        """
        logger.info(f"[PCA] Aplicando PCA con n_components={self.n_components}")

        self.pca = PCA(n_components=self.n_components, random_state=42)
        principal_components = self.pca.fit_transform(returns_df)

        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.components = self.pca.components_

        # Crear DataFrame con componentes
        pc_df = pd.DataFrame(
            principal_components,
            index=returns_df.index,
            columns=[f'PC{i+1}' for i in range(principal_components.shape[1])]
        )

        total_variance_explained = self.explained_variance_ratio.sum()

        logger.info(
            f"[PCA] Reducido de {len(returns_df.columns)} activos a "
            f"{len(pc_df.columns)} componentes principales"
        )
        logger.info(f"[PCA] Varianza explicada: {total_variance_explained:.2%}")

        return pc_df

    def calculate_denoised_covariance(
        self,
        returns_df: pd.DataFrame,
        frequency: int = 252
    ) -> pd.DataFrame:
        """
        Calcula matriz de covarianza "limpia" usando PCA.

        Proceso:
        1. Aplicar PCA a retornos
        2. Reconstruir retornos usando solo componentes principales
        3. Calcular covarianza sobre retornos reconstruidos

        Args:
            returns_df: Retornos diarios
            frequency: Frecuencia para anualizar (252 para días)

        Returns:
            Matriz de covarianza denoised anualizada
        """
        logger.info("[PCA] Calculando matriz de covarianza denoised...")

        # Aplicar PCA
        pc_df = self.fit_transform(returns_df)

        # Reconstruir retornos (inversión de PCA)
        reconstructed = self.pca.inverse_transform(pc_df.values)
        reconstructed_df = pd.DataFrame(
            reconstructed,
            index=returns_df.index,
            columns=returns_df.columns
        )

        # Covarianza sobre datos reconstruidos
        cov_matrix = reconstructed_df.cov() * frequency

        # Calcular métricas de calidad
        reconstruction_error = np.mean((returns_df.values - reconstructed) ** 2)
        logger.info(f"[PCA] Error de reconstrucción: {reconstruction_error:.6f}")

        return cov_matrix

    def get_factor_exposures(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula exposición de cada activo a cada componente principal.

        Args:
            returns_df: Retornos diarios

        Returns:
            DataFrame con shape (n_assets, n_components)
            - Filas: tickers
            - Columnas: PC1, PC2, ...
            - Valores: loadings (correlación activo-componente)
        """
        if self.pca is None:
            self.fit_transform(returns_df)

        # Loadings = correlación entre activos y componentes
        # components_.T * sqrt(explained_variance)
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)

        loadings_df = pd.DataFrame(
            loadings,
            index=returns_df.columns,
            columns=[f'PC{i+1}' for i in range(loadings.shape[1])]
        )

        logger.info(f"[PCA] Factor exposures calculados para {len(loadings_df)} activos")

        return loadings_df

    def explain_components(
        self,
        returns_df: pd.DataFrame,
        top_n_assets: int = 5
    ) -> pd.DataFrame:
        """
        Explica cada componente principal identificando los activos más influyentes.

        Args:
            returns_df: Retornos diarios
            top_n_assets: Número de activos top a mostrar por componente

        Returns:
            DataFrame con interpretación de cada componente
        """
        if self.pca is None:
            self.fit_transform(returns_df)

        loadings_df = self.get_factor_exposures(returns_df)

        explanations = []

        for pc in loadings_df.columns:
            # Obtener top activos por loading (valor absoluto)
            abs_loadings = loadings_df[pc].abs().sort_values(ascending=False)
            top_assets = abs_loadings.head(top_n_assets)

            explanation = {
                'component': pc,
                'variance_explained': self.explained_variance_ratio[int(pc[2:])-1],
                'top_assets': top_assets.index.tolist(),
                'loadings': top_assets.values.tolist()
            }

            explanations.append(explanation)

        explanations_df = pd.DataFrame(explanations)

        return explanations_df

    def get_risk_contribution(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray
    ) -> pd.DataFrame:
        """
        Calcula contribución al riesgo de cada componente principal.

        Args:
            returns_df: Retornos diarios
            weights: Pesos del portafolio

        Returns:
            DataFrame con contribución de riesgo por componente
        """
        if self.pca is None:
            self.fit_transform(returns_df)

        # Transformar pesos al espacio de componentes principales
        weights_pc = self.pca.transform(weights.reshape(1, -1))[0]

        # Varianza de cada componente
        component_variances = self.pca.explained_variance_

        # Contribución al riesgo = peso^2 * varianza
        risk_contributions = (weights_pc ** 2) * component_variances

        # Normalizar a porcentaje
        total_risk = risk_contributions.sum()
        risk_pct = risk_contributions / total_risk if total_risk > 0 else risk_contributions

        risk_df = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(len(risk_contributions))],
            'risk_contribution': risk_contributions,
            'risk_pct': risk_pct,
            'explained_variance': self.pca.explained_variance_ratio_
        })

        return risk_df

    def to_dict(self) -> dict:
        """Serializa información del modelo PCA"""
        if self.pca is None:
            return {'fitted': False}

        return {
            'fitted': True,
            'n_components_input': self.n_components,
            'n_components_actual': self.pca.n_components_,
            'explained_variance_ratio': self.explained_variance_ratio.tolist(),
            'total_variance_explained': float(self.explained_variance_ratio.sum())
        }
