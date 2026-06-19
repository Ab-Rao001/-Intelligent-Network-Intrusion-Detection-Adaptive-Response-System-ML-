"""
cluster_model.py — Unsupervised Learning Module

Provides:
  - PCA dimensionality reduction (configurable n_components)
  - K-Means clustering with elbow method support
  - Silhouette score evaluation
  - Model persistence (save/load)
"""

import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.logger import app_logger


class ClusteringPipeline:
    """PCA + K-Means clustering pipeline for network traffic analysis."""

    def __init__(self, n_pca_components: int = 2, n_clusters: int = 5, random_state: int = 42):
        self.n_pca_components = n_pca_components
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.pca = PCA(n_components=n_pca_components, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

        self.pca_data: np.ndarray | None = None
        self.cluster_labels: np.ndarray | None = None
        self.silhouette: float | None = None

    # ──────────────────────────────────────────
    # PCA
    # ──────────────────────────────────────────
    def fit_pca(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and return transformed data."""
        try:
            self.pca_data = self.pca.fit_transform(X)
            app_logger.info(f"PCA fit successful. Transformed shape: {self.pca_data.shape}")
            return self.pca_data
        except ValueError as ve:
            app_logger.error(f"ValueError during PCA fitting (shape mismatch or NaN): {str(ve)}", exc_info=True)
            raise ValueError(f"Failed to fit PCA due to invalid data format: {str(ve)}")
        except Exception as e:
            app_logger.error(f"Unexpected error during PCA fitting: {str(e)}", exc_info=True)
            raise RuntimeError(f"Unexpected error in PCA: {str(e)}")

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        """Transform new data with fitted PCA."""
        return self.pca.transform(X)

    def get_explained_variance(self) -> np.ndarray:
        """Return explained variance ratio per component."""
        return self.pca.explained_variance_ratio_

    # ──────────────────────────────────────────
    # K-Means
    # ──────────────────────────────────────────
    def fit_kmeans(self, X: np.ndarray) -> np.ndarray:
        """Fit K-Means on the (PCA-reduced) data."""
        try:
            self.cluster_labels = self.kmeans.fit_predict(X)
            if len(np.unique(self.cluster_labels)) > 1:
                self.silhouette = silhouette_score(X, self.cluster_labels)
            else:
                self.silhouette = -1.0
            app_logger.info(f"K-Means fit successful. Silhouette: {self.silhouette:.4f}")
            return self.cluster_labels
        except MemoryError as me:
            app_logger.error(f"MemoryError during K-Means fitting: {str(me)}", exc_info=True)
            raise MemoryError("System ran out of memory computing K-Means/Silhouette score. Try sampling down the dataset.")
        except ValueError as ve:
            app_logger.error(f"ValueError during K-Means fitting: {str(ve)}", exc_info=True)
            raise ValueError(f"Failed to fit K-Means due to data shape or content: {str(ve)}")
        except Exception as e:
            app_logger.error(f"Unexpected error during K-Means fitting: {str(e)}", exc_info=True)
            raise RuntimeError(f"Unexpected error in K-Means: {str(e)}")

    @staticmethod
    def elbow_method(X: np.ndarray, k_range: range = range(2, 11), random_state: int = 42):
        """Compute inertia for each k — used for the elbow plot.

        Returns:
            list of (k, inertia) tuples
        """
        results = []
        try:
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                km.fit(X)
                results.append((k, km.inertia_))
            app_logger.info("Elbow method computation completed successfully.")
            return results
        except MemoryError as me:
            app_logger.error(f"MemoryError during Elbow Method: {str(me)}", exc_info=True)
            raise MemoryError("System ran out of memory computing Elbow Curve. Try a smaller sample.")
        except Exception as e:
            app_logger.error(f"Error during Elbow Method: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to compute Elbow Curve: {str(e)}")

    # ──────────────────────────────────────────
    # Full Pipeline
    # ──────────────────────────────────────────
    def run(self, X: np.ndarray):
        """Run PCA → K-Means pipeline.

        Returns:
            pca_data, cluster_labels, silhouette_score
        """
        pca_data = self.fit_pca(X)
        labels = self.fit_kmeans(pca_data)
        return pca_data, labels, self.silhouette

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────
    def save(self, directory: str = "saved_models"):
        """Save PCA and K-Means models."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.pca, os.path.join(directory, "pca_model.pkl"))
        joblib.dump(self.kmeans, os.path.join(directory, "kmeans_model.pkl"))
        print(f"Clustering models saved to {directory}/")

    def load(self, directory: str = "saved_models"):
        """Load PCA and K-Means models."""
        self.pca = joblib.load(os.path.join(directory, "pca_model.pkl"))
        self.kmeans = joblib.load(os.path.join(directory, "kmeans_model.pkl"))


if __name__ == "__main__":
    # Quick test with random data
    X_demo = np.random.rand(500, 20)
    pipeline = ClusteringPipeline(n_pca_components=2, n_clusters=5)
    pca_out, labels, sil = pipeline.run(X_demo)
    print(f"PCA shape: {pca_out.shape}")
    print(f"Cluster labels: {np.unique(labels)}")
    print(f"Silhouette Score: {sil:.4f}")
    pipeline.save()
