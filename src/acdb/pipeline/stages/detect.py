"""Stage 2 - ``detect``: run the configured detector over every extracted frame.

The detector itself is pluggable (``detect.type``); this stage adds the
*dataset-level* gates from ``detect.filter`` (confidence, minimum side, aspect
ratio, relative area, per-frame cap) via
:func:`acdb.detectors.filters.filter_detections`, and - optionally - a
cross-frame NMS that suppresses a subject which barely moved since the previous
sampled frame, which is the cheapest way to stop interval sampling from
producing dozens of near-identical crops of a standing character.

Output: ``detections.csv`` (raw detector geometry, one row per accepted box).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from acdb.detectors import build_detector, filter_detections
from acdb.detectors.base import Detection
from acdb.detectors.filters import box_iou
from acdb.errors import StageError
from acdb.pipeline.context import StageContext, StageOutcome, reusable_items
from acdb.pipeline.stage import Stage
from acdb.utils.device import describe_device, resolve_device
from acdb.utils.image import imread

__all__ = ["DetectStage"]


class DetectStage(Stage):
    name = "detect"
    section = "detect"
    description = "detect characters/persons in extracted frames"

    def execute(self, ctx: StageContext) -> StageOutcome:
        cfg = ctx.cfg
        frames = ctx.load("frames", required=["frame_id", "video_id", "frame_path"])
        if frames.empty:
            raise StageError("frames.csv is empty - run `acdb extract-frames` first")

        done = reusable_items(ctx.resume_items, frames["frame_id"].astype(str))
        carried = ctx.carry_over("detections", "frame_id", done)
        rows: List[dict] = carried.to_dict("records") if not carried.empty else []

        pending = frames.loc[~frames["frame_id"].astype(str).isin(done)] if done else frames
        device = resolve_device(cfg.project.device)
        detector = build_detector(cfg.detect, device)
        ctx.info(f"detector: {detector.describe()} on {describe_device(device)}")

        processed = filtered = produced = 0
        items: List[str] = []
        reasons: Dict[str, int] = {}
        per_frame: Dict[str, List[Detection]] = {}
        previous: Dict[str, List[Tuple[int, Detection]]] = {}

        try:
            # Frames are processed video by video so decode order stays sequential.
            for _video_id, group in pending.groupby("video_id", sort=False):
                batch: List[Tuple[dict, object]] = []
                for record in group.sort_values("frame_idx").to_dict("records"):
                    try:
                        image = imread(ctx.rel_abs(str(record["frame_path"])))
                    except StageError as exc:
                        ctx.warning(f"skipping {record['frame_path']}: {exc}")
                        continue
                    batch.append((record, image))
                    if len(batch) >= max(1, cfg.detect.batch_size):
                        processed, filtered, produced = self._flush(
                            ctx, detector, batch, rows, reasons, previous, per_frame,
                            processed, filtered, produced, items,
                        )
                        batch = []
                if batch:
                    processed, filtered, produced = self._flush(
                        ctx, detector, batch, rows, reasons, previous, per_frame,
                        processed, filtered, produced, items,
                    )
        finally:
            detector.close()

        if not rows:
            raise StageError(
                "no detection survived the filters. Loosen detect.filter "
                "(min_conf / min_side / min_area_ratio) or check your weights."
            )

        frame = pd.DataFrame(rows)
        if not ctx.dry_run:
            ctx.save("detections", frame)
        per_frame_counts = frame.groupby("frame_id").size() if not frame.empty else pd.Series(dtype=int)
        empty_frames = int((~frames["frame_id"].isin(per_frame_counts.index)).sum())
        ctx.info(
            f"detections.csv: {len(frame)} box(es) over "
            f"{per_frame_counts.size} frame(s); {empty_frames} frame(s) empty"
        )
        extra = {
            "detector": cfg.detect.type,
            "device": device,
            "empty_frames": empty_frames,
        }
        if reasons:
            extra["rejected"] = ",".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
        return StageOutcome(
            processed=processed,
            filtered=filtered,
            produced=produced,
            items=items,
            outputs=[str(ctx.table("detections"))],
            extra=extra,
        )

    # -- helpers ----------------------------------------------------------- #
    def _flush(
        self,
        ctx: StageContext,
        detector,
        batch: Sequence[Tuple[dict, object]],
        rows: List[dict],
        reasons: Dict[str, int],
        previous: Dict[str, List[Tuple[int, Detection]]],
        per_frame: Dict[str, List[Detection]],
        processed: int,
        filtered: int,
        produced: int,
        items: List[str],
    ) -> Tuple[int, int, int]:
        frames = [image for _record, image in batch]
        results = detector.detect_batch(frames)
        for (record, image), detections in zip(batch, results):
            frame_id = str(record["frame_id"])
            keep, rejected = filter_detections(
                detections, ctx.cfg.detect.filter, image.shape[:2]
            )
            for reason, count in rejected.items():
                reasons[reason] = reasons.get(reason, 0) + count
            filtered += sum(rejected.values())

            if ctx.cfg.detect.filter.nms_across_frames:
                video_id = str(record["video_id"])
                keep = _suppress_unchanged(
                    keep, previous.get(video_id, []), threshold=0.9
                )
            previous[str(record["video_id"])] = [
                (int(record["frame_idx"]), detection) for detection in keep
            ]

            per_frame[frame_id] = keep
            for bbox_idx, detection in enumerate(keep):
                rows.append(
                    {
                        "frame_id": frame_id,
                        "video_path": record["video_path"],
                        "video_id": record["video_id"],
                        "frame_idx": int(record["frame_idx"]),
                        "timestamp_ms": int(record.get("timestamp_ms", 0)),
                        "strategy": record.get("strategy", ""),
                        "frame_path": record["frame_path"],
                        "bbox_idx": int(bbox_idx),
                        "bbox_x1": int(detection.bbox[0]),
                        "bbox_y1": int(detection.bbox[1]),
                        "bbox_x2": int(detection.bbox[2]),
                        "bbox_y2": int(detection.bbox[3]),
                        "det_conf": round(float(detection.confidence), 6),
                        "class_name": str(detection.class_name),
                        "width": int(image.shape[1]),
                        "height": int(image.shape[0]),
                    }
                )
            processed += 1
            produced += len(keep)
            items.append(frame_id)
            ctx.mark_processed(frame_id)
        return processed, filtered, produced


def _suppress_unchanged(
    current: Sequence[Detection],
    previous: Sequence[Tuple[int, Detection]],
    *,
    threshold: float = 0.9,
) -> List[Detection]:
    """Drop boxes that overlap a detection of the previous sampled frame."""
    if not previous:
        return list(current)
    kept: List[Detection] = []
    for detection in current:
        if any(
            box_iou(detection.bbox, earlier.bbox) >= threshold
            for _frame_idx, earlier in previous
        ):
            continue
        kept.append(detection)
    return kept
