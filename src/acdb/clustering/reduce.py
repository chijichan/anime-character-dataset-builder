"""Dimensionality reduction used before clustering.

PCA is on by default because every distance-based clusterer degrades in high
dimensions (the curse of dimensionality makes cosine/Euclidean neighbourhoods
meaningless), and embedding spaces are frequently 512-1024-D. Reduction is
cosmetic-free: it is deterministic (``svd_solver="full"``) so two runs on the
same data produce the same clusters.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from acdb.utils.logging import get_logger

__all__ = ["reduce_features"]


def reduce_features(
    features: np.ndarray,
    method: str = "pca",
    components: int = 32,
    *,
    seed: int = 1337,
) -> Tuple[np.ndarray, dict]:
    """Return ``(reduced, info)``; ``method='none'`` passes the input through."""
    matrix = np.asarray(features, dtype=np.float64)
    name = (method or "none").strip().lower()
    info: dict = {"method": name, "input_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0}

    if name in {"none", "null", ""} or matrix.ndim != 2 or matrix.shape[0] < 3:
        info["output_dim"] = info["input_dim"]
        return np.asarray(features, dtype=np.float32), info

    if name != "pca":
        get_logger("cluster").warning(f"unknown cluster.reduce_dim={method!r}; skipping reduction")
        info["method"] = "none"
        info["output_dim"] = info["input_dim"]
        return np.asarray(features, dtype=np.float32), info

    from sklearn.decomposition import PCA

    max_components = max(1, min(matrix.shape[0] - 1, matrix.shape[1]))
    k = int(max(2, min(int(components), max_components)))
    if k >= matrix.shape[1]:
        info["output_dim"] = info["input_dim"]
        info["explained_variance"] = 1.0
        return np.asarray(features, dtype=np.float32), info

    model = PCA(n_components=k, svd_solver="full", random_state=seed)
    reduced = model.fit_transform(matrix).astype(np.float32)
    info["output_dim"] = int(k)
    info["explained_variance"] = float(np.sum(model.explained_variance_ratio_))
    return reduced, info
