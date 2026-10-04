"""Embedding cache.

``dedup`` (cosine refinement) and ``cluster`` need the same embeddings, and a
re-run after a tweak to the *clustering* parameters must not re-run the
backbone. Embeddings are therefore cached in ``data/metadata/features.npz``,
keyed by image id and guarded by an extractor signature: change the model (or
its parameters) and the cache is transparently rebuilt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from acdb.extractors.base import BaseFeatureExtractor
from acdb.utils.atomic import temp_sibling
from acdb.utils.logging import get_logger

__all__ = ["FeatureSet", "compute_features", "load_feature_cache"]


@dataclass
class FeatureSet:
    """Image ids plus their embedding matrix (row-aligned)."""

    ids: List[str]
    vectors: np.ndarray
    signature: str = ""

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def index(self) -> Dict[str, int]:
        return {image_id: position for position, image_id in enumerate(self.ids)}

    def row(self, image_id: str) -> Optional[np.ndarray]:
        position = self.index().get(image_id)
        return None if position is None else self.vectors[position]


def load_feature_cache(path: Path, signature: str) -> Optional[FeatureSet]:
    """Load a cache file when it exists *and* matches ``signature``."""
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        with np.load(file_path, allow_pickle=False) as payload:
            stored = str(payload["signature"]) if "signature" in payload else ""
            if stored != signature:
                get_logger("features").info(
                    "feature cache signature changed; recomputing all embeddings"
                )
                return None
            ids = [str(v) for v in payload["ids"].tolist()]
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        get_logger("features").warning(f"ignoring unreadable feature cache {file_path}: {exc}")
        return None
    if len(ids) != len(vectors):
        return None
    return FeatureSet(ids=ids, vectors=vectors, signature=signature)


def save_feature_cache(path: Path, feature_set: FeatureSet) -> Path:
    """Atomically persist a :class:`FeatureSet`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = temp_sibling(target)
    try:
        with open(staging, "wb") as handle:
            np.savez(
                handle,
                # Stored as a fixed-width unicode array so the cache can be read
                # back with allow_pickle=False (an object array cannot).
                ids=np.asarray(feature_set.ids, dtype=str),
                vectors=feature_set.vectors.astype(np.float32),
                signature=np.asarray(feature_set.signature),
            )
        os.replace(staging, target)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return target


def compute_features(
    image_ids: Sequence[str],
    loader: Callable[[str], np.ndarray],
    extractor: BaseFeatureExtractor,
    *,
    cache_path: Optional[Path] = None,
    batch_size: int = 32,
    progress: bool = True,
) -> FeatureSet:
    """Return embeddings for ``image_ids``, reusing/refreshing the on-disk cache.

    ``loader`` maps an image id to a BGR ``uint8`` array. Only cache misses are
    decoded and embedded, so an interrupted run resumes cheaply.
    """
    ids = [str(image_id) for image_id in image_ids]
    signature = extractor.signature()
    logger = get_logger("features")

    cached: Dict[str, np.ndarray] = {}
    if cache_path is not None:
        existing = load_feature_cache(Path(cache_path), signature)
        if existing is not None:
            position = existing.index()
            cached = {
                image_id: existing.vectors[position[image_id]]
                for image_id in ids
                if image_id in position
            }
            if cached:
                logger.info(f"feature cache hit for {len(cached)}/{len(ids)} images")

    missing = [image_id for image_id in ids if image_id not in cached]
    if missing:
        logger.info(
            f"embedding {len(missing)} image(s) with {extractor.describe()}"
        )
        iterator = range(0, len(missing), max(1, batch_size))
        if progress:  # pragma: no cover - cosmetic
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="features", unit="batch", leave=False)
            except ImportError:
                pass
        for start in iterator:
            chunk = missing[start : start + max(1, batch_size)]
            images = [loader(image_id) for image_id in chunk]
            vectors = extractor.extract(images)
            for image_id, vector in zip(chunk, vectors):
                cached[image_id] = vector.astype(np.float32)

    vectors = np.stack([cached[image_id] for image_id in ids]).astype(np.float32) if ids else np.zeros((0, 0), np.float32)
    feature_set = FeatureSet(ids=ids, vectors=vectors, signature=signature)
    if cache_path is not None:
        save_feature_cache(Path(cache_path), feature_set)
        logger.debug(f"feature cache written to {cache_path}")
    return feature_set
