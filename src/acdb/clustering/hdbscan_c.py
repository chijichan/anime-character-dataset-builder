"""HDBSCAN clusterer (sklearn implementation).

Preferred backend: it finds the number of clusters itself, tolerates the
non-spherical clusters that colour/embedding spaces produce, and - crucially for
this pipeline - labels outliers as noise instead of forcing every crop into a
character. Those noise points become ``unknown/`` and are reviewed by hand.

Falls back to :class:`~acdb.clustering.kmeans.KMeansClusterer` semantics only
when the caller asks for it; a missing scikit-learn is a hard error because
scikit-learn is a core dependency.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from acdb.clustering.base import BaseClusterer, ClusterResult
from acdb.errors import StageError

__all__ = ["HDBSCANClusterer"]


class HDBSCANClusterer(BaseClusterer):
    """``sklearn.cluster.HDBSCAN`` wrapper (density-based, variable k)."""

    name = "hdbscan"

    def __init__(self, params: Mapping[str, Any], *, seed: int = 1337) -> None:
        super().__init__(params, seed=seed)
        try:  # pragma: no cover - dependency is core
            from sklearn.cluster import HDBSCAN
        except ImportError as exc:  # pragma: no cover
            raise StageError(
                "sklearn.cluster.HDBSCAN requires scikit-learn >= 1.3 "
                "(pip install -U scikit-learn), or set cluster.type=kmeans"
            ) from exc
        self._hdbscan_cls = HDBSCAN

    def fit_predict(self, features: np.ndarray) -> ClusterResult:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return ClusterResult(labels=np.zeros(0, dtype=int), backend=self.name)

        min_cluster_size = int(self.param("min_cluster_size", 4))
        # With very few samples, HDBSCAN would call everything noise; shrinking
        # the size keeps small demo/CI runs meaningful.
        min_cluster_size = max(2, min(min_cluster_size, max(2, matrix.shape[0] // 2)))
        min_samples = self.param("min_samples", None)
        min_samples = None if min_samples is None else max(1, int(min_samples))

        model = self._hdbscan_cls(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=str(self.param("metric", "euclidean")),
            cluster_selection_epsilon=float(self.param("cluster_selection_epsilon", 0.0)),
            cluster_selection_method=str(self.param("cluster_selection_method", "eom")),
            alpha=float(self.param("alpha", 1.0)),
            allow_single_cluster=bool(self.param("allow_single_cluster", False)),
            store_centers=None,
            copy=True,
        )
        labels = np.asarray(model.fit_predict(matrix), dtype=int)
        probabilities = getattr(model, "probabilities_", None)
        return ClusterResult(
            labels=labels,
            probabilities=None if probabilities is None else np.asarray(probabilities, dtype=float),
            backend=self.name,
            params={
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "metric": str(self.param("metric", "euclidean")),
            },
        )
