"""Feature-extractor abstraction.

An extractor maps a batch of BGR crops to an ``(N, D)`` float32 embedding
matrix. Both :mod:`acdb.pipeline.stages.dedup` (cosine refinement) and
:mod:`acdb.pipeline.stages.cluster` consume it, which is why the embeddings are
cached on disk keyed by image content.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = ["BaseFeatureExtractor", "l2_normalize"]


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation (safe for zero rows)."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


class BaseFeatureExtractor(ABC):
    """Interface every feature extractor implements."""

    name: str = "base"
    #: Embedding dimensionality when known up front (``None`` = decided at runtime).
    dim: Optional[int] = None

    def __init__(self, params: Mapping[str, Any], device: str = "cpu", *, normalize: bool = True) -> None:
        self.params: Mapping[str, Any] = dict(params or {})
        self.device = device
        self.normalize_output = normalize

    # -- API --------------------------------------------------------------- #
    @abstractmethod
    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Embed a batch of BGR ``uint8`` images (H, W, 3)."""

    def extract_one(self, image: np.ndarray) -> np.ndarray:
        return self.extract([image])[0]

    def warmup(self) -> None:
        """Optional first-call initialisation (download/shape inference)."""

    def close(self) -> None:
        """Release resources (no-op by default)."""

    def describe(self) -> str:
        return f"{self.name}(dim={self.dim}, device={self.device})"

    def signature(self) -> str:
        """Stable identity of the extractor configuration (for the cache key)."""
        from acdb.config import hash_mapping

        return hash_mapping(
            {"type": self.name, "params": dict(self.params), "normalize": self.normalize_output}
        )

    # -- helpers ----------------------------------------------------------- #
    def param(self, key: str, default: Any = None) -> Any:
        value = self.params.get(key, default)
        return default if value is None else value

    def _finalize(self, vectors: np.ndarray) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if self.normalize_output:
            array = l2_normalize(array)
        if self.dim is None:
            self.dim = int(array.shape[1])
        return array
