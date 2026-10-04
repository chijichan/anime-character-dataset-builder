"""The nine pipeline stages (one module each)."""

from __future__ import annotations

from acdb.pipeline.stages.apply_labels import ApplyLabelsStage
from acdb.pipeline.stages.cluster import ClusterStage
from acdb.pipeline.stages.crop import CropStage
from acdb.pipeline.stages.dedup import DedupStage
from acdb.pipeline.stages.detect import DetectStage
from acdb.pipeline.stages.export import ExportStage
from acdb.pipeline.stages.extract_frames import ExtractFramesStage
from acdb.pipeline.stages.filter import FilterStage
from acdb.pipeline.stages.split import SplitStage

__all__ = [
    "ExtractFramesStage",
    "DetectStage",
    "CropStage",
    "DedupStage",
    "FilterStage",
    "ClusterStage",
    "ApplyLabelsStage",
    "SplitStage",
    "ExportStage",
    "STAGE_CLASSES",
]

STAGE_CLASSES = {
    "extract_frames": ExtractFramesStage,
    "detect": DetectStage,
    "crop": CropStage,
    "dedup": DedupStage,
    "filter": FilterStage,
    "cluster": ClusterStage,
    "apply_labels": ApplyLabelsStage,
    "split": SplitStage,
    "export": ExportStage,
}
