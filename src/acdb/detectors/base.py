"""Detector abstraction.

A detector turns one BGR frame into a list of :class:`Detection`. Everything
downstream (crop, dedup, filter, cluster) only depends on this base class, so
swapping YOLOv8 for an anime-specific checkpoint - or for the dependency-free
saliency detector - is a one-line configuration change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Detection", "BaseDetector"]

BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    """One detected subject in *source-frame* pixel coordinates."""

    bbox: BBox
    confidence: float
    class_id: int = 0
    class_name: str = "person"
    #: Optional boolean mask (``True`` = subject) with the frame's H/W.
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    def as_dict(self) -> dict:
        return {
            "bbox_x1": int(self.bbox[0]),
            "bbox_y1": int(self.bbox[1]),
            "bbox_x2": int(self.bbox[2]),
            "bbox_y2": int(self.bbox[3]),
            "det_conf": round(float(self.confidence), 6),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }


class BaseDetector(ABC):
    """Interface every detector implements."""

    #: Short identifier used in logs and state files.
    name: str = "base"

    def __init__(self, params: Mapping[str, Any], device: str = "cpu") -> None:
        self.params: Mapping[str, Any] = dict(params or {})
        self.device = device

    # -- API --------------------------------------------------------------- #
    @abstractmethod
    def detect(self, image: np.ndarray) -> Sequence[Detection]:
        """Return detections for a single BGR frame."""

    def detect_batch(self, images: Sequence[np.ndarray]) -> Sequence[Sequence[Detection]]:
        """Default: loop :meth:`detect`. Batched backends override this."""
        return [self.detect(image) for image in images]

    def warmup(self) -> None:
        """Optional first-call initialisation (weights, CUDA context, ...)."""

    def close(self) -> None:
        """Release resources (no-op by default)."""

    def describe(self) -> str:
        return f"{self.name}(device={self.device})"

    # -- helpers ----------------------------------------------------------- #
    def param(self, key: str, default: Any = None) -> Any:
        value = self.params.get(key, default)
        return default if value is None else value
