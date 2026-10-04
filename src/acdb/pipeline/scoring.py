"""Quality scoring, shared by ``filter`` and ``dedup``.

``filter`` applies the gates; ``dedup`` needs the *same* score to decide which
member of a duplicate group survives (``dedup.keep: best_quality``), so the
scoring lives in one module used by both.

The composite score is a weighted mean of four normalised, monotone components.
Each component is mapped onto ``[0, 1]`` against a fixed reference so that the
number is comparable across videos and runs:

=============  ==========================================================
sharpness      ``log1p(var(Laplacian)) / log1p(1000)`` - the log keeps
               3000-var frames from saturating the scale
exposure       ``1 - max(over/limit, under/limit)`` - clipping is the only
               kind of exposure damage that is unrecoverable
contrast       ``std(gray) / 64``
colorfulness   ``Hasler-Susstrunk / 60``
=============  ==========================================================

Hard gates (blur, minimum side, clipping ratio, colorfulness floor, minimum
score) stay in the ``filter`` stage; :func:`composite_score` deliberately does
not gate anything so ``dedup`` can rank even images the filter would drop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from acdb.config import QualityConfig
from acdb.utils.image import quality_metrics

__all__ = [
    "QualityVerdict",
    "composite_score",
    "evaluate",
    "score_metrics",
    "SHARPNESS_REF",
    "CONTRAST_REF",
    "COLORFULNESS_REF",
]

SHARPNESS_REF = 1000.0
CONTRAST_REF = 64.0
COLORFULNESS_REF = 60.0

#: Metric columns persisted into ``quality.csv`` / ``metadata.csv``.
METRIC_COLUMNS: Sequence[str] = (
    "laplacian_var",
    "overexposed_ratio",
    "underexposed_ratio",
    "colorfulness",
    "contrast",
    "brightness",
)


@dataclass(frozen=True)
class QualityVerdict:
    score: float
    metrics: Dict[str, float]
    passed: bool
    reasons: Tuple[str, ...] = ()


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _sharpness(laplacian_var: float) -> float:
    return _clip01(math.log1p(max(0.0, laplacian_var)) / math.log1p(SHARPNESS_REF))


def _exposure(over: float, under: float, max_over: float, max_under: float) -> float:
    worst = max(
        over / max_over if max_over > 0 else 0.0,
        under / max_under if max_under > 0 else 0.0,
    )
    return _clip01(1.0 - worst)


def _contrast(std: float) -> float:
    return _clip01(std / CONTRAST_REF)


def _colorfulness(value: float) -> float:
    return _clip01(value / COLORFULNESS_REF)


def _weights(cfg: QualityConfig) -> Dict[str, float]:
    raw = {
        "sharpness": float(cfg.weights.sharpness),
        "exposure": float(cfg.weights.exposure),
        "contrast": float(cfg.weights.contrast),
        "colorfulness": float(cfg.weights.colorfulness),
    }
    total = sum(raw.values())
    if total <= 0:  # pragma: no cover - validation rejects this
        return {key: 0.25 for key in raw}
    return {key: value / total for key, value in raw.items()}


def composite_score(metrics: Mapping[str, float], cfg: QualityConfig) -> float:
    """Weighted composite in ``[0, 1]`` (no gating)."""
    weights = _weights(cfg)
    components = {
        "sharpness": _sharpness(float(metrics.get("laplacian_var", 0.0))),
        "exposure": _exposure(
            float(metrics.get("overexposed_ratio", 0.0)),
            float(metrics.get("underexposed_ratio", 0.0)),
            float(cfg.max_overexposed_ratio),
            float(cfg.max_underexposed_ratio),
        ),
        "contrast": _contrast(float(metrics.get("contrast", 0.0))),
        "colorfulness": _colorfulness(float(metrics.get("colorfulness", 0.0))),
    }
    return float(sum(weights[name] * value for name, value in components.items()))


def score_metrics(metrics: Mapping[str, float], cfg: QualityConfig) -> float:
    """Alias of :func:`composite_score` used where metrics are already computed."""
    return composite_score(metrics, cfg)


def _gate_reasons(metrics: Mapping[str, float], score: float, cfg: QualityConfig) -> Tuple[str, ...]:
    reasons = []
    if cfg.enabled is False:
        return ()
    if float(metrics.get("laplacian_var", 0.0)) < cfg.min_laplacian_var:
        reasons.append("blurry")
    if min(float(metrics.get("width", 0.0)), float(metrics.get("height", 0.0))) < cfg.min_side_px:
        reasons.append("too_small")
    if float(metrics.get("overexposed_ratio", 0.0)) > cfg.max_overexposed_ratio:
        reasons.append("overexposed")
    if float(metrics.get("underexposed_ratio", 0.0)) > cfg.max_underexposed_ratio:
        reasons.append("underexposed")
    if float(metrics.get("colorfulness", 0.0)) < cfg.min_colorfulness:
        reasons.append("low_colorfulness")
    if score < cfg.min_score:
        reasons.append("low_score")
    return tuple(reasons)


def evaluate(image: np.ndarray, cfg: QualityConfig) -> QualityVerdict:
    """Measure ``image`` and apply the gates."""
    metrics = quality_metrics(image)
    score = composite_score(metrics, cfg)
    reasons = _gate_reasons(metrics, score, cfg)
    return QualityVerdict(score=score, metrics=metrics, passed=not reasons, reasons=reasons)
