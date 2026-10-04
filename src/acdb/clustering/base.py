"""Clusterer abstraction.

A clusterer maps an ``(N, D)`` embedding matrix to an integer label per row.
``-1`` always means *noise / unassigned* (the HDBSCAN convention, reused by the
other backends so downstream code has a single rule to follow).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

__all__ = ["ClusterResult", "BaseClusterer"]

NOISE_LABEL = -1


@dataclass
class ClusterResult:
    """Labels plus optional per-point membership strength."""

    labels: np.ndarray
    probabilities: Optional[np.ndarray] = None
    backend: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return int(len({int(v) for v in self.labels if int(v) >= 0}))

    @property
    def n_noise(self) -> int:
        return int(np.count_nonzero(self.labels < 0))

    def sizes(self) -> dict:
        counts: dict = {}
        for label in self.labels:
            counts[int(label)] = counts.get(int(label), 0) + 1
        return dict(sorted(counts.items()))

    def label_of(self, index: int) -> int:
        return int(self.labels[index])


class BaseClusterer(ABC):
    """Interface every clusterer implements."""

    name: str = "base"

    def __init__(self, params: Mapping[str, Any], *, seed: int = 1337) -> None:
        self.params: Mapping[str, Any] = dict(params or {})
        self.seed = int(seed)

    @abstractmethod
    def fit_predict(self, features: np.ndarray) -> ClusterResult:
        """Cluster ``features`` (N, D) and return a :class:`ClusterResult`."""

    def describe(self) -> str:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({rendered})"

    def param(self, key: str, default: Any = None) -> Any:
        value = self.params.get(key, default)
        return default if value is None else value
