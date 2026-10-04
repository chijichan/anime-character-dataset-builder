"""Dependency-free classical detector: border-referenced saliency + contours.

Why this exists
---------------
The repository ships **no** model weights (they are copyrighted and/or licence
encumbered), and downloading a checkpoint on first run would make the demo
non-reproducible offline. This detector gives the pipeline a fully working,
dependency-free ``detect.type`` that behaves like a real detector: it returns
boxes *and* masks, with confidences derived from a measurable saliency score.

How it works
------------
1. Estimate the background colour from a border ring (median in CIE-Lab). The
   border of a shot is background in the overwhelming majority of anime frames
   (characters are centred, and the frame edge is scenery or letterboxing).
2. Threshold the Lab distance to that background colour -> a saliency mask.
3. Clean it morphologically, fill holes, and take external contours.
4. Emit a box per contour, with confidence mapped from the mean saliency inside
   the contour through ``conf_from_saliency = [lo, hi]``.

It is a *baseline*, not a replacement for a trained detector: it finds the
salient subject, not "the character". That is enough for the synthetic demo
video and for quick smoke tests on stylised footage.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from acdb.detectors.base import BaseDetector, Detection
from acdb.utils.image import saliency_distance

__all__ = ["SaliencyContourDetector"]


class SaliencyContourDetector(BaseDetector):
    """Salient-region detector (see the module docstring)."""

    name = "saliency"

    def __init__(self, params: Mapping[str, Any], device: str = "cpu") -> None:
        super().__init__(params, device)
        self.min_distance = float(self.param("min_distance", 26.0))
        self.border_frac = float(self.param("border_frac", 0.08))
        self.blur_ksize = int(self.param("blur_ksize", 5))
        self.morph_ksize = int(self.param("morph_ksize", 7))
        self.min_area_px = int(self.param("min_area_px", 400))
        self.max_boxes = int(self.param("max_boxes", 8))
        self.min_fill_ratio = float(self.param("min_fill_ratio", 0.15))
        conf_range = self.param("conf_from_saliency", [0.30, 0.95])
        self.conf_low, self.conf_high = float(conf_range[0]), float(conf_range[1])
        names = list(self.param("class_names", ["character"]) or ["character"])
        self.class_names = names

    # -- API --------------------------------------------------------------- #
    def describe(self) -> str:
        return (
            f"saliency(min_distance={self.min_distance}, min_area_px={self.min_area_px}, "
            f"border_frac={self.border_frac})"
        )

    def detect(self, image: np.ndarray) -> Sequence[Detection]:
        if image.ndim != 3 or image.shape[0] < 8 or image.shape[1] < 8:
            return []

        distance = saliency_distance(
            image, border_frac=self.border_frac, blur_ksize=self.blur_ksize
        )
        mask = (distance >= self.min_distance).astype(np.uint8) * 255
        if self.morph_ksize >= 3:
            ksize = self.morph_ksize if self.morph_ksize % 2 else self.morph_ksize + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 1 or h <= 1:
                continue
            if area / float(w * h) < self.min_fill_ratio:
                continue
            region = distance[y : y + h, x : x + w]
            mean_distance = float(region.mean()) if region.size else 0.0
            detections.append(
                Detection(
                    bbox=(int(x), int(y), int(x + w), int(y + h)),
                    confidence=self._confidence(mean_distance),
                    class_id=0,
                    class_name=self.class_names[0],
                    mask=self._contour_mask(mask.shape, contour),
                )
            )

        detections.sort(key=lambda d: -d.confidence)
        return detections[: self.max_boxes]

    # -- internals --------------------------------------------------------- #
    def _confidence(self, mean_distance: float) -> float:
        span = max(1e-6, self.min_distance)
        ratio = float(np.clip((mean_distance - span) / (4.0 * span), 0.0, 1.0))
        return float(self.conf_low + ratio * (self.conf_high - self.conf_low))

    @staticmethod
    def _contour_mask(shape: Tuple[int, int], contour: np.ndarray) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        return mask.astype(bool)
