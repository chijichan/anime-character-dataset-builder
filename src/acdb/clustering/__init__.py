"""Clusterer registry - ``cluster.type`` in the config selects the backend."""

from __future__ import annotations

from typing import Dict, List, Mapping, Type

from acdb.clustering.base import NOISE_LABEL, BaseClusterer, ClusterResult
from acdb.clustering.hdbscan_c import HDBSCANClusterer
from acdb.clustering.kmeans_c import AgglomerativeClusterer, KMeansClusterer
from acdb.clustering.reduce import reduce_features
from acdb.config import ClusterConfig
from acdb.errors import ConfigError

__all__ = [
    "CLUSTERER_TYPES",
    "BaseClusterer",
    "ClusterResult",
    "NOISE_LABEL",
    "HDBSCANClusterer",
    "KMeansClusterer",
    "AgglomerativeClusterer",
    "build_clusterer",
    "available_clusterers",
    "reduce_features",
]

CLUSTERER_TYPES: Dict[str, Type[BaseClusterer]] = {
    "hdbscan": HDBSCANClusterer,
    "kmeans": KMeansClusterer,
    "agglomerative": AgglomerativeClusterer,
}


def available_clusterers() -> List[str]:
    return sorted(CLUSTERER_TYPES)


def build_clusterer(config: ClusterConfig, *, seed: int = 1337) -> BaseClusterer:
    """Instantiate the clusterer named by ``config.type``."""
    name = (config.type or "").strip().lower()
    factory: Mapping[str, Type[BaseClusterer]] | Type[BaseClusterer] = CLUSTERER_TYPES.get(name)  # type: ignore[assignment]
    if factory is None:
        raise ConfigError(
            f"unknown cluster.type {config.type!r}. Available: {available_clusterers()}"
        )
    return factory(dict(config.params or {}), seed=seed)  # type: ignore[operator]
