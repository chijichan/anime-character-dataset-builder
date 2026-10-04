"""Ultralytics YOLO detector (YOLOv8 / YOLO11 / anime-specific checkpoints).

One implementation covers every YOLO flavour: the differences are the weights
path plus a handful of parameters, all of which come from the config. The
``type`` aliases are resolved in :mod:`acdb.detectors`:

``yolo_v8``        ``yolov8n.pt`` (COCO person) - the safe default
``yolo11``         ``yolo11n.pt``
``anime_yolo``     a single-class anime character checkpoint (bring your own)
``anime_yolo_seg`` a segmentation checkpoint; masks enable
                   ``crop.keep_background: false``

No weights are bundled with this repository. Point ``detect.params.weights`` at
a local ``.pt`` file, or leave ``model`` at an official ultralytics name and let
ultralytics download it on first use (see the README for anime checkpoint
pointers).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import numpy as np

from acdb.detectors.base import BaseDetector, Detection
from acdb.errors import ModelUnavailableError, StageError

# Ultralytics imports matplotlib for its plotting helpers, which in an
# interactive session prints "Backend tkagg is interactive backend. Turning
# interactive mode on." and can open windows. This pipeline never plots, so ask
# matplotlib for the non-interactive backend *before* ultralytics is imported.
# setdefault() keeps an explicit user choice intact.
os.environ.setdefault("MPLBACKEND", "Agg")

__all__ = ["YoloDetector"]

_INSTALL_HINT = (
    "ultralytics is not installed. Install the optional extra with\n"
    "    pip install -e \".[yolo]\"\n"
    "or run with detect.type=saliency for the dependency-free detector."
)


class YoloDetector(BaseDetector):
    """Thin, config-driven wrapper around ``ultralytics.YOLO``."""

    name = "yolo"

    def __init__(self, params: Mapping[str, Any], device: str = "cpu", *, alias: str = "yolo_v8") -> None:
        super().__init__(params, device)
        self.alias = alias

        try:  # pragma: no cover - optional dependency
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelUnavailableError(_INSTALL_HINT) from exc

        weights = self.param("weights") or self.param("model")
        if not weights:
            raise ModelUnavailableError(
                f"detect.params.weights/model is empty for type={alias}. "
                "Set an ultralytics model name (e.g. yolov8n.pt) or a local .pt path."
            )
        weights = str(weights)
        if weights.lower().endswith(".pt") and Path(weights).is_absolute() and not Path(weights).is_file():
            raise ModelUnavailableError(f"weight file not found: {weights}")

        self.weights = weights
        self.conf = float(self.param("conf_threshold", 0.35))
        self.iou = float(self.param("iou_threshold", 0.5))
        self.imgsz = int(self.param("imgsz", 640))
        self.max_det = int(self.param("max_detections", 20))
        self.half = bool(self.param("half", False)) and device.startswith("cuda")
        self.agnostic_nms = bool(self.param("agnostic_nms", False))
        self.retina_masks = bool(self.param("retina_masks", False))
        self.classes = self.param("classes", None)
        self.class_names = list(self.param("class_names", []) or [])
        self.want_masks = bool(self.param("masks", alias.endswith("_seg")))

        try:  # pragma: no cover - optional dependency
            self._model = YOLO(weights)
        except Exception as exc:  # pragma: no cover - download/format errors
            raise ModelUnavailableError(
                f"cannot load YOLO weights {weights!r}: {exc}\n"
                "Check the path, or download an anime character checkpoint manually "
                "(see README -> Model weights)."
            ) from exc

        if self.classes is not None and not isinstance(self.classes, (list, tuple)):
            self.classes = [int(self.classes)]

    # -- API --------------------------------------------------------------- #
    def describe(self) -> str:
        return f"{self.alias}(weights={self.weights}, device={self.device}, conf={self.conf})"

    def detect(self, image: np.ndarray) -> Sequence[Detection]:
        return self.detect_batch([image])[0]

    def detect_batch(self, images: Sequence[np.ndarray]) -> Sequence[Sequence[Detection]]:
        if not images:
            return []
        kwargs = {
            "source": list(images),
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "classes": self.classes,
            "device": self.device,
            "agnostic_nms": self.agnostic_nms,
            "retina_masks": self.retina_masks,
            "verbose": False,
        }
        kwargs.update(self._precision_kwargs())
        try:  # pragma: no cover - optional dependency
            results = self._model.predict(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime/backend errors
            raise StageError(f"{self.alias} inference failed: {exc}") from exc

        batch: List[List[Detection]] = []
        for result in results:
            batch.append(self._parse_result(result))
        return batch

    def _precision_kwargs(self) -> Mapping[str, Any]:
        """The fp16 switch, spelled the way this ultralytics version expects.

        ultralytics >= 8.4 replaced ``half``/``int8`` with a unified
        ``quantize`` argument and warns about the legacy ones on every call.
        We therefore omit the argument entirely for fp32 (its default) and only
        pass fp16 in the spelling the installed version understands.
        """
        if not self.half:
            return {}
        try:
            from ultralytics.cfg import QUANTIZE_ALIASES  # noqa: F401

            return {"quantize": 16}  # 16 == fp16 in the unified scheme
        except ImportError:  # pragma: no cover - older ultralytics
            return {"half": True}

    # -- parsing ----------------------------------------------------------- #
    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return str(self.class_names[class_id])
        names = getattr(self._model, "names", None) or {}
        if isinstance(names, Mapping) and class_id in names:
            return str(names[class_id])
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return f"class_{class_id}"

    def _parse_result(self, result: Any) -> List[Detection]:  # pragma: no cover - needs weights
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        masks = None
        if self.want_masks and getattr(result, "masks", None) is not None:
            try:
                masks = result.masks.data.cpu().numpy().astype(bool)
            except Exception:
                masks = None

        detections: List[Detection] = []
        for index in range(len(xyxy)):
            x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy[index][:4])
            mask = masks[index] if masks is not None and index < len(masks) else None
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidences[index]),
                    class_id=int(classes[index]),
                    class_name=self._class_name(int(classes[index])),
                    mask=mask,
                )
            )
        return detections
