"""Quality scoring, geometry helpers and detection filtering."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from acdb.config import DetectFilterConfig, QualityConfig
from acdb.detectors.base import Detection
from acdb.detectors.filters import box_iou, filter_detections
from acdb.pipeline.scoring import composite_score, evaluate
from acdb.utils.image import (
    apply_background_mask,
    colorfulness,
    crop_region,
    expand_bbox,
    exposure_ratios,
    laplacian_variance,
    mean_abs_diff,
    pad_to_square,
    quality_metrics,
    resize_long_side,
)

# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_expand_bbox_scales_about_the_centre() -> None:
    assert expand_bbox((10, 20, 30, 40), 2.0) == (0.0, 10.0, 40.0, 50.0)
    assert expand_bbox((0, 0, 10, 10), 1.0) == (0.0, 0.0, 10.0, 10.0)


def test_expand_bbox_squarifies() -> None:
    x1, y1, x2, y2 = expand_bbox((0, 0, 10, 30), 1.0, square=True)
    assert pytest.approx(x2 - x1) == y2 - y1 == 30
    assert pytest.approx((x1 + x2) / 2) == 5


def test_crop_region_returns_realised_rect_and_handles_edges() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[40:60, 50:80] = 255

    crop, rect = crop_region(image, (50, 40, 80, 60), expand_ratio=1.0, square=False)
    assert rect == (50, 40, 80, 60)
    assert crop.shape[:2] == (20, 30)

    # A box crossing the frame border is padded, never wrapped or crashed.
    padded, rect = crop_region(image, (0, 0, 20, 20), expand_ratio=2.0, square=True)
    assert padded.shape[0] == padded.shape[1] == 40
    assert rect[0] < 0


def test_pad_to_square_and_resize() -> None:
    image = np.zeros((10, 30, 3), dtype=np.uint8)
    square = pad_to_square(image)
    assert square.shape[:2] == (30, 30)
    assert resize_long_side(image, 15).shape[:2] == (5, 15)
    assert resize_long_side(image, 99) is image  # never upscales
    assert resize_long_side(image, None) is image


def test_apply_background_mask() -> None:
    image = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    masked = apply_background_mask(image, mask, (0, 0, 0))
    assert masked[0, 0].tolist() == [0, 0, 0]
    assert masked[1, 1].tolist() == [200, 200, 200]
    with pytest.raises(ValueError):
        apply_background_mask(image, np.zeros((2, 2), dtype=bool), (0, 0, 0))


# --------------------------------------------------------------------------- #
# Quality metrics (pure)
# --------------------------------------------------------------------------- #


def _sharp_image() -> np.ndarray:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    for index in range(0, 64, 4):
        image[index : index + 2, :] = 255
    return image


def _blurred_image() -> np.ndarray:
    return cv2.GaussianBlur(_sharp_image(), (31, 31), 0)


def test_laplacian_variance_separates_sharp_from_blurred() -> None:
    assert laplacian_variance(_sharp_image()) > 100 * laplacian_variance(_blurred_image())


def test_exposure_ratios() -> None:
    white = np.full((10, 10, 3), 255, dtype=np.uint8)
    black = np.zeros((10, 10, 3), dtype=np.uint8)
    over, under = exposure_ratios(white)
    assert over == 1.0 and under == 0.0
    over, under = exposure_ratios(black)
    assert over == 0.0 and under == 1.0


def test_colorfulness_of_grey_is_zero() -> None:
    grey = np.full((16, 16, 3), 128, dtype=np.uint8)
    assert colorfulness(grey) == pytest.approx(0.0, abs=1e-6)
    colourful = np.zeros((16, 16, 3), dtype=np.uint8)
    colourful[:, :8] = (255, 0, 0)
    colourful[:, 8:] = (0, 0, 255)
    assert colorfulness(colourful) > 20


def test_quality_metrics_shape() -> None:
    metrics = quality_metrics(_sharp_image())
    for key in ("laplacian_var", "overexposed_ratio", "underexposed_ratio",
                "colorfulness", "contrast", "brightness", "width", "height"):
        assert key in metrics
    assert metrics["width"] == 64 and metrics["height"] == 64


def test_mean_abs_diff() -> None:
    first = np.zeros((4, 4, 3), dtype=np.uint8)
    second = np.full((4, 4, 3), 10, dtype=np.uint8)
    assert mean_abs_diff(first, second) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        mean_abs_diff(first, np.zeros((2, 2, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Composite score and gates
# --------------------------------------------------------------------------- #


def test_composite_score_is_higher_for_sharper_images() -> None:
    cfg = QualityConfig()
    sharp = composite_score(quality_metrics(_sharp_image()), cfg)
    blurred = composite_score(quality_metrics(_blurred_image()), cfg)
    assert sharp > blurred


def test_composite_score_respects_weights() -> None:
    cfg = QualityConfig(min_laplacian_var=0.0, min_side_px=0, min_score=0.0)
    metrics = {"laplacian_var": 1000.0, "overexposed_ratio": 0.0, "underexposed_ratio": 0.0,
               "contrast": 255.0, "colorfulness": 200.0}
    perfect = composite_score(metrics, cfg)
    assert perfect == pytest.approx(1.0, abs=1e-6)
    zeroed = composite_score({**metrics, "laplacian_var": 0.0}, cfg)
    assert zeroed < perfect


def test_evaluate_applies_every_gate() -> None:
    cfg = QualityConfig(min_laplacian_var=1e9)
    verdict = evaluate(_sharp_image(), cfg)
    assert not verdict.passed and "blurry" in verdict.reasons

    cfg = QualityConfig(min_side_px=512, min_laplacian_var=0.0, min_colorfulness=0.0,
                        max_overexposed_ratio=1.0, max_underexposed_ratio=1.0, min_score=0.0)
    verdict = evaluate(_sharp_image(), cfg)
    assert not verdict.passed and "too_small" in verdict.reasons

    cfg = QualityConfig(max_overexposed_ratio=0.01, min_laplacian_var=0.0, min_side_px=0,
                        min_colorfulness=0.0, min_score=0.0)
    verdict = evaluate(_sharp_image(), cfg)
    assert not verdict.passed and "overexposed" in verdict.reasons

    cfg = QualityConfig(min_colorfulness=500.0, min_laplacian_var=0.0, min_side_px=0,
                        max_overexposed_ratio=1.0, max_underexposed_ratio=1.0, min_score=0.0)
    verdict = evaluate(_sharp_image(), cfg)
    assert not verdict.passed and "low_colorfulness" in verdict.reasons

    cfg = QualityConfig(min_score=0.99, min_laplacian_var=0.0, min_side_px=0,
                        min_colorfulness=0.0, max_overexposed_ratio=1.0,
                        max_underexposed_ratio=1.0)
    verdict = evaluate(_blurred_image(), cfg)
    assert not verdict.passed and "low_score" in verdict.reasons

    cfg = QualityConfig(enabled=False, min_laplacian_var=1e9)
    assert evaluate(_blurred_image(), cfg).passed, "disabled quality must not gate"


# --------------------------------------------------------------------------- #
# Detector filtering
# --------------------------------------------------------------------------- #


def _detection(x1, y1, x2, y2, confidence=0.9, name="person") -> Detection:
    return Detection(bbox=(x1, y1, x2, y2), confidence=confidence, class_name=name)


def test_filter_detections_applies_each_gate() -> None:
    cfg = DetectFilterConfig(min_conf=0.5, min_side=20, aspect_ratio=(0.5, 2.0),
                             min_area_ratio=0.0, max_per_frame=10)
    frame_shape = (100, 100)
    kept, rejected = filter_detections(
        [
            _detection(0, 0, 40, 40, 0.9),      # keep
            _detection(0, 0, 40, 40, 0.1),      # below_min_conf
            _detection(0, 0, 10, 40, 0.9),      # too_small (short side 10)
            _detection(0, 0, 90, 20, 0.9),      # bad_aspect_ratio (4.5 > 2.0)
            _detection(-10, -10, 30, 30, 0.9),  # clamped to 30x30, then kept
        ],
        cfg,
        frame_shape,
    )
    assert len(kept) == 2
    assert rejected["below_min_conf"] == 1
    assert rejected["too_small"] == 1
    assert rejected["bad_aspect_ratio"] == 1
    assert kept[0].confidence >= kept[1].confidence, "kept boxes are confidence-ordered"
    assert all(box.bbox[0] >= 0 and box.bbox[2] <= 100 for box in kept)


def test_filter_detections_cap_and_area_ratio() -> None:
    cfg = DetectFilterConfig(min_conf=0.0, min_side=1, aspect_ratio=(0.01, 100.0),
                             min_area_ratio=0.05, max_per_frame=2)
    boxes = [_detection(0, 0, 2, 2, 0.9) for _ in range(4)]  # area ratio 0.0004
    kept, rejected = filter_detections(boxes, cfg, (100, 100))
    assert kept == [] and rejected["too_small_area"] == 4

    big = [_detection(0, 0, 50, 50, 0.5 + index * 0.1) for index in range(4)]
    kept, rejected = filter_detections(big, cfg, (100, 100))
    assert len(kept) == 2 and rejected["over_max_per_frame"] == 2
    assert kept[0].confidence > kept[1].confidence


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_filter_detections_can_drop_the_subtitle_band() -> None:
    """Burned-in subtitles sit at the bottom; characters may overlap that band."""
    cfg = DetectFilterConfig(min_conf=0.0, min_side=1, aspect_ratio=(0.01, 100.0),
                             min_area_ratio=0.0, max_per_frame=10, ignore_bottom_frac=0.2)
    subtitle = _detection(100, 940, 800, 990)      # centred at y=965 of 1000 -> dropped
    character = _detection(200, 100, 700, 980)     # centred at y=540 -> kept
    kept, rejected = filter_detections([subtitle, character], cfg, (1000, 1000))
    assert [box.bbox for box in kept] == [(200, 100, 700, 980)]
    assert rejected == {"subtitle_band": 1}

    # Disabled by default: nothing is dropped for the wrong reason.
    permissive = DetectFilterConfig(min_conf=0.0, min_side=1, aspect_ratio=(0.01, 100.0),
                                    min_area_ratio=0.0)
    assert len(filter_detections([subtitle], permissive, (1000, 1000)).kept) == 1
