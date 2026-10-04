"""KMeans and Agglomerative clusterers - the fixed-k alternatives to HDBSCAN.

Both are useful when you already know how many characters to expect, or when
HDBSCAN drowns everything in noise on a small dataset. There is no noise
concept: every sample gets a cluster (``noise_as_unknown`` then has no effect).
"""

from __future__ import annotations

import numpy as np

from acdb.clustering.base import BaseClusterer, ClusterResult
from acdb.errors import StageError

__all__ = ["KMeansClusterer", "AgglomerativeClusterer"]


def _effective_k(requested: int, n_samples: int) -> int:
    if n_samples <= 0:
        return 0
    return max(1, min(int(requested), n_samples))


class KMeansClusterer(BaseClusterer):
    """``sklearn.cluster.KMeans`` (spherical, fixed k, seeded)."""

    name = "kmeans"

    def fit_predict(self, features: np.ndarray) -> ClusterResult:
        from sklearn.cluster import KMeans

        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return ClusterResult(labels=np.zeros(0, dtype=int), backend=self.name)
        k = _effective_k(int(self.param("n_clusters", 12)), matrix.shape[0])
        if k <= 1:
            return ClusterResult(labels=np.zeros(len(matrix), dtype=int), backend=self.name)

        model = KMeans(
            n_clusters=k,
            n_init=int(self.param("n_init", 10)),
            max_iter=int(self.param("max_iter", 300)),
            tol=float(self.param("tol", 1e-4)),
            random_state=self.seed,
        )
        labels = np.asarray(model.fit_predict(matrix), dtype=int)
        return ClusterResult(
            labels=labels,
            probabilities=None,
            backend=self.name,
            params={"n_clusters": k},
        )


class AgglomerativeClusterer(BaseClusterer):
    """``sklearn.cluster.AgglomerativeClustering`` (hierarchical, fixed k)."""

    name = "agglomerative"

    def fit_predict(self, features: np.ndarray) -> ClusterResult:
        from sklearn.cluster import AgglomerativeClustering

        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return ClusterResult(labels=np.zeros(0, dtype=int), backend=self.name)
        k = _effective_k(int(self.param("n_clusters", 12)), matrix.shape[0])
        if k <= 1:
            return ClusterResult(labels=np.zeros(len(matrix), dtype=int), backend=self.name)

        linkage = str(self.param("linkage", "ward"))
        metric = str(self.param("metric", "euclidean"))
        if linkage == "ward" and metric != "euclidean":
            raise StageError("agglomerative linkage='ward' requires metric='euclidean'")
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric=metric,
        )
        labels = np.asarray(model.fit_predict(matrix), dtype=int)
        return ClusterResult(
            labels=labels,
            probabilities=None,
            backend=self.name,
            params={"n_clusters": k, "linkage": linkage},
        )
