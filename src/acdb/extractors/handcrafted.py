"""Hand-crafted HSV histogram embeddings - no weights, no optional extras.

Why this exists: it is the only extractor that runs in a clean environment with
nothing but numpy/OpenCV installed, which makes the demo reproducible offline.
For flat-shaded anime art it is also genuinely competitive at *identity*
separation, because a character's palette (hair, uniform, accessories) is
usually the dominant cue::

    features:
      type: color_hist
      params: {color_bins: 8, spatial_grid: 2, include_gradient: true}

Layout of the ``272``-dimensional vector (defaults):
``spatial_grid^2 * color_bins^2`` joint hue/saturation cell histograms, then an
8-bin gradient-orientation histogram, then per-cell V mean/std.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

import cv2
import numpy as np

from acdb.extractors.base import BaseFeatureExtractor

__all__ = ["ColorHistogramExtractor"]


class ColorHistogramExtractor(BaseFeatureExtractor):
    """Spatial colour/gradient descriptor (see the module docstring)."""

    name = "color_hist"

    def __init__(self, params: Mapping[str, Any], device: str = "cpu", *, normalize: bool = True) -> None:
        super().__init__(params, device, normalize=normalize)
        self.color_bins = int(self.param("color_bins", 8))
        self.spatial_grid = int(self.param("spatial_grid", 2))
        self.include_gradient = bool(self.param("include_gradient", True))
        self.resize = int(self.param("resize", 128))
        self.gradient_bins = int(self.param("gradient_bins", 8))
        self.dim = self._compute_dim()

    def _compute_dim(self) -> int:
        cells = max(1, self.spatial_grid) ** 2
        dim = cells * self.color_bins * self.color_bins + 2 * cells
        if self.include_gradient:
            dim += self.gradient_bins
        return dim

    # -- API --------------------------------------------------------------- #
    def describe(self) -> str:
        return (
            f"color_hist(bins={self.color_bins}, grid={self.spatial_grid}, "
            f"gradient={self.include_gradient}, dim={self.dim})"
        )

    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.stack([self._describe_image(image) for image in images]).astype(np.float32)
        return self._finalize(vectors)

    # -- internals --------------------------------------------------------- #
    def _describe_image(self, image: np.ndarray) -> np.ndarray:
        size = max(32, self.resize)
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        grid = max(1, self.spatial_grid)
        cell = size // grid
        bins = self.color_bins
        features: List[float] = []

        for row in range(grid):
            for column in range(grid):
                y0, x0 = row * cell, column * cell
                patch = hsv[y0 : y0 + cell, x0 : x0 + cell]
                if patch.size == 0:
                    features.extend([0.0] * (bins * bins + 2))
                    continue
                histogram = cv2.calcHist(
                    [patch], [0, 1], None, [bins, bins], [0, 180, 0, 256]
                ).reshape(-1)
                total = float(histogram.sum()) or 1.0
                features.extend((histogram / total).tolist())
                features.append(float(patch[:, :, 2].mean()) / 255.0)
                features.append(float(patch[:, :, 2].std()) / 255.0)

        if self.include_gradient:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(gx, gy)
            angle = (np.arctan2(gy, gx) + np.pi) % np.pi  # unsigned orientation
            histogram, _ = np.histogram(
                angle, bins=self.gradient_bins, range=(0.0, np.pi), weights=magnitude
            )
            total = float(histogram.sum()) or 1.0
            features.extend((histogram / total).tolist())

        return np.asarray(features, dtype=np.float32)
