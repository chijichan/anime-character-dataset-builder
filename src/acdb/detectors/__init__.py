"""Detector registry - ``detect.type`` in the config selects the implementation.

Adding a detector means writing a :class:`~acdb.detectors.base.BaseDetector`
subclass and registering it in :data:`DETECTOR_TYPES`; no pipeline code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Type

from acdb.config import DetectConfig
from acdb.detectors.base import BaseDetector, Detection
from acdb.detectors.filters import FilterOutcome, filter_detections
from acdb.detectors.saliency import SaliencyContourDetector
from acdb.detectors.yolo import YoloDetector
from acdb.errors import ConfigError

__all__ = [
    "DETECTOR_TYPES",
    "DetectorSpec",
    "BaseDetector",
    "Detection",
    "FilterOutcome",
    "filter_detections",
    "build_detector",
    "available_detectors",
]


@dataclass(frozen=True)
class DetectorSpec:
    """Registry entry: implementation + per-alias default parameters."""

    factory: Type[BaseDetector]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    requires: str = ""


DETECTOR_TYPES: Dict[str, DetectorSpec] = {
    "yolo_v8": DetectorSpec(
        YoloDetector,
        {"model": "yolov8n.pt", "class_names": ["person"], "classes": [0]},
        "Ultralytics YOLOv8 COCO detector (person class by default).",
        requires="ultralytics",
    ),
    "yolo11": DetectorSpec(
        YoloDetector,
        {"model": "yolo11n.pt", "class_names": ["person"], "classes": [0]},
        "Ultralytics YOLO11 COCO detector.",
        requires="ultralytics",
    ),
    "anime_yolo": DetectorSpec(
        YoloDetector,
        {"class_names": ["character"], "classes": [0]},
        "Anime character checkpoint (bring your own weights; see README).",
        requires="ultralytics + local .pt",
    ),
    "anime_yolo_seg": DetectorSpec(
        YoloDetector,
        {"class_names": ["character"], "classes": [0], "masks": True},
        "Anime character segmentation checkpoint; masks enable background removal.",
        requires="ultralytics + local .pt",
    ),
    "saliency": DetectorSpec(
        SaliencyContourDetector,
        {},
        "Classical border-referenced saliency detector; no weights, no extras.",
        requires="none",
    ),
}


def available_detectors() -> List[str]:
    return sorted(DETECTOR_TYPES)


def build_detector(config: DetectConfig, device: str = "cpu") -> BaseDetector:
    """Instantiate the detector named by ``config.type``."""
    name = (config.type or "").strip()
    spec = DETECTOR_TYPES.get(name)
    if spec is None:
        raise ConfigError(
            f"unknown detect.type {name!r}. Available: {available_detectors()}"
        )
    params: Dict[str, Any] = dict(spec.defaults)
    params.update(dict(config.params or {}))
    if spec.factory is YoloDetector:
        return YoloDetector(params, device, alias=name)
    return spec.factory(params, device)
