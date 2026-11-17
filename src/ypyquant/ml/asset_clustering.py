"""
Servicio de clustering de activos para pre-filtrado del universo.
Reduce el número de activos seleccionando representantes de cada cluster.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

logger = logging.getLogger(__name__)


class AssetClusteringService:
    """
    Pre-filtrado de activos usando clustering para mejorar diversificación.

    Selecciona representantes de cada cluster para reducir el universo
    manteniendo máxima diversificación.
    """

    def __init__(self, method: str = 'hierarchical'):
        """
        Args:
            method: 'hierarchical', 'kmeans'
        """
        self.method = method
        self.cluster_labels = None
        self.cluster_info = {}

    def select_representative_assets(
        self,
        returns_df: pd.DataFrame,
        n_clusters: int = 5,
        selection_method: str = 'max_sharpe'
    ) -> list[str]:
        """
        Selecciona activos representativos de cada cluster.

        Args:
            returns_df: Retornos históricos diarios
            n_clusters: Número de clusters a formar
            selection_method: Cómo seleccionar representante:
                - 'max_sharpe': Mejor Sharpe de cada cluster
                - 'centroid': Más cercano al centroide
                - 'min_correlation': Menor correlación promedio

        Returns:
            Lista de tickers seleccionados (uno por cluster)
        """
        if len(returns_df.columns) <= n_clusters:
            logger.warning(
                f"[Clustering] Número de activos ({len(returns_df.columns)}) "
                f"<= clusters ({n_clusters}), retornando todos los activos"
            )
            return list(returns_df.columns)

        logger.info(
            f"[Clustering] Iniciando clustering con método={self.method}, "
            f"n_clusters={n_clusters}, selection={selection_method}"
        )

        # Calcular matriz de correlación
        corr_matrix = returns_df.corr()
        distance_matrix = 1 - corr_matrix.abs()

        # Clustering
        if self.method == 'hierarchical':
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)

        elif self.method == 'kmeans':
            # KMeans sobre matriz de correlación (necesita array 2D)
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clustering.fit_predict(corr_matrix)

        else:
            raise ValueError(f"Método de clustering desconocido: {self.method}")

        self.cluster_labels = labels

        # Seleccionar representante por cluster
        selected = []

        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_tickers = returns_df.columns[cluster_mask].tolist()

            if len(cluster_tickers) == 0:
                continue

            if len(cluster_tickers) == 1:
                selected.append(cluster_tickers[0])
                self.cluster_info[cluster_id] = {
                    'tickers': cluster_tickers,
                    'selected': cluster_tickers[0],
                    'size': 1
                }
                continue

            # Seleccionar representante según método
            if selection_method == 'max_sharpe':
                # Seleccionar el de mejor Sharpe
                cluster_returns = returns_df[cluster_tickers]
                mean_ret = cluster_returns.mean() * 252
                vol = cluster_returns.std() * np.sqrt(252)
                sharpe = mean_ret / vol
                sharpe = sharpe.replace([np.inf, -np.inf], 0).fillna(0)
                best = sharpe.idxmax()
                selected.append(best)

            elif selection_method == 'centroid':
                # Seleccionar el más cercano al centroide del cluster
                cluster_corr = corr_matrix.loc[cluster_tickers, cluster_tickers]
                avg_corr = cluster_corr.mean(axis=1)
                # El más correlacionado con todos = más central
                best = avg_corr.idxmax()
                selected.append(best)

            elif selection_method == 'min_correlation':
                # Seleccionar el MENOS correlacionado (más único)
                cluster_corr = corr_matrix.loc[cluster_tickers, cluster_tickers]
                avg_corr = cluster_corr.mean(axis=1)
                best = avg_corr.idxmin()
                selected.append(best)

            else:
                raise ValueError(f"Método de selección desconocido: {selection_method}")

            # Guardar info del cluster
            self.cluster_info[cluster_id] = {
                'tickers': cluster_tickers,
                'selected': best,
                'size': len(cluster_tickers)
            }

        logger.info(
            f"[Clustering] Seleccionados {len(selected)} activos de {len(returns_df.columns)} "
            f"({len(selected)/len(returns_df.columns)*100:.1f}% reducción)"
        )

        return selected

    def get_cluster_summary(self) -> dict:
        """
        Retorna resumen de los clusters formados.

        Returns:
            Dict con información de cada cluster
        """
        if not self.cluster_info:
            return {"error": "No clustering performed yet"}

        summary = {
            'total_clusters': len(self.cluster_info),
            'clusters': self.cluster_info,
            'total_assets': sum(info['size'] for info in self.cluster_info.values()),
            'selected_assets': [info['selected'] for info in self.cluster_info.values()]
        }

        return summary

    def visualize_clusters(
        self,
        returns_df: pd.DataFrame,
        save_path: str | None = None
    ):
        """
        Visualiza los clusters formados (requiere matplotlib).

        Args:
            returns_df: Retornos históricos
            save_path: Path para guardar imagen (opcional)
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet. Call select_representative_assets() first.")

        try:
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import dendrogram, linkage

            # Crear matriz de correlación
            corr_matrix = returns_df.corr()
            distance_matrix = 1 - corr_matrix.abs()

            # Crear dendrogram si es hierarchical
            if self.method == 'hierarchical':
                linkage_matrix = linkage(distance_matrix, method='average')

                plt.figure(figsize=(12, 8))
                dendrogram(
                    linkage_matrix,
                    labels=returns_df.columns.tolist(),
                    leaf_rotation=90
                )
                plt.title('Hierarchical Clustering Dendrogram')
                plt.xlabel('Ticker')
                plt.ylabel('Distance (1 - |correlation|)')
                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path)
                    logger.info(f"[Clustering] Dendrogram guardado en {save_path}")
                else:
                    plt.show()

        except ImportError:
            logger.warning("[Clustering] matplotlib no disponible para visualización")
