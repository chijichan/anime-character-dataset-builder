"""Post-detection filtering - pure functions, unit-tested.

The detector's own confidence threshold is a *model* knob; the rules here are
*dataset* knobs (how small a character may be, how extreme its aspect ratio may
get, how crowded a frame may be). Keeping them separate means you can loosen the
model threshold to find rare poses without polluting the dataset.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence, Tuple

from acdb.config import DetectFilterConfig
from acdb.detectors.base import Detection

__all__ = ["filter_detections", "FilterOutcome", "box_iou"]

_REASONS = (
    "below_min_conf",
    "too_small",
    "too_large",
    "bad_aspect_ratio",
    "too_small_area",
    "subtitle_band",
    "over_max_per_frame",
)


class FilterOutcome(NamedTuple):
    """``(kept, rejected_by_reason)`` - unpacks like a 2-tuple."""

    kept: List[Detection]
    rejected: Dict[str, int]

    @property
    def total_rejected(self) -> int:
        return sum(self.rejected.values())


def filter_detections(
    detections: Sequence[Detection],
    config: DetectFilterConfig,
    frame_shape: Tuple[int, int],
) -> FilterOutcome:
    """Apply the configured geometric/confidence gates.

    ``frame_shape`` is ``(height, width)`` of the source frame, used for the
    relative-area gate. Surviving boxes are clamped to the frame and sorted by
    descending confidence, then truncated to ``max_per_frame``.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    frame_area = max(1, height * width)
    rejected: Dict[str, int] = {reason: 0 for reason in _REASONS}
    kept: List[Detection] = []

    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            rejected["too_small"] += 1
            continue

        box = Detection(
            bbox=(x1, y1, x2, y2),
            confidence=float(detection.confidence),
            class_id=int(detection.class_id),
            class_name=str(detection.class_name),
            mask=detection.mask,
        )

        if box.confidence < config.min_conf:
            rejected["below_min_conf"] += 1
            continue
        short_side, long_side = min(box.width, box.height), max(box.width, box.height)
        if short_side < config.min_side:
            rejected["too_small"] += 1
            continue
        if config.max_side is not None and long_side > config.max_side:
            rejected["too_large"] += 1
            continue
        low, high = config.aspect_ratio
        if not (low <= box.aspect_ratio <= high):
            rejected["bad_aspect_ratio"] += 1
            continue
        if box.area / frame_area < config.min_area_ratio:
            rejected["too_small_area"] += 1
            continue
        if config.ignore_bottom_frac > 0:
            band_top = (1.0 - min(0.9, config.ignore_bottom_frac)) * height
            if (y1 + y2) / 2 >= band_top:
                rejected["subtitle_band"] += 1
                continue
        kept.append(box)

    kept.sort(key=lambda d: (-d.confidence, d.bbox))
    if config.max_per_frame and len(kept) > config.max_per_frame:
        rejected["over_max_per_frame"] += len(kept) - config.max_per_frame
        kept = kept[: config.max_per_frame]

    return FilterOutcome(kept, {k: v for k, v in rejected.items() if v})


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in first)
    bx1, by1, bx2, by2 = (float(v) for v in second)
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
