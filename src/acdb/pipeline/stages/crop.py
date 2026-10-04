"""Stage 3 - ``crop``: turn every detection into a square-ish image file.

* ``expand_ratio`` grows the box about its centre (default 1.2 == 20% context).
* ``square: true`` pads the short side so every crop has the same aspect ratio,
  which is what embedding backbones expect.
* ``keep_background: false`` neutralises the background with
  ``background_fill``, using a per-crop border-ring saliency estimate (see the
  README's design decisions for the caveat).
* The perceptual hash is computed here, while the pixels are already in memory,
  so ``dedup`` never has to re-read a file.

The ``metadata.csv`` bounding-box columns describe the **realised crop
rectangle** in source-frame coordinates rather than the raw detector box, so
every exported image stays reproducible from the manifest alone. Raw detector
geometry is preserved in ``detections.csv``.

Output: ``data/crops/{video_id}/{video_id}_f{frame_idx:06d}_b{bbox_idx:02d}.jpg``
plus ``crops.csv``.
"""

from __future__ import annotations

from typing import List

import cv2
import pandas as pd

from acdb.errors import StageError
from acdb.io import build_image_id
from acdb.pipeline.context import StageContext, StageOutcome, reusable_items
from acdb.pipeline.stage import Stage
from acdb.utils.image import (
    apply_background_mask,
    crop_region,
    imread,
    imwrite,
    perception_hash,
    saliency_distance,
)

__all__ = ["CropStage"]


class CropStage(Stage):
    name = "crop"
    section = "crop"
    description = "crop detections, expand to a square and hash them"

    def execute(self, ctx: StageContext) -> StageOutcome:
        cfg = ctx.cfg
        settings = cfg.crop
        # Writing 72 crops is cheap; checkpoint every 50 to bound the loss.
        ctx.flush_every = 50
        detections = ctx.load(
            "detections", required=["video_id", "frame_idx", "bbox_idx", "frame_path"]
        )
        if detections.empty:
            raise StageError("detections.csv is empty - run `acdb detect` first")

        detections = detections.copy()
        detections["image_id"] = [
            build_image_id(video_id, frame_idx, bbox_idx)
            for video_id, frame_idx, bbox_idx in zip(
                detections["video_id"], detections["frame_idx"], detections["bbox_idx"]
            )
        ]

        done = reusable_items(ctx.resume_items, detections["image_id"].astype(str))
        carried = ctx.carry_over("crops", "image_id", done)
        rows: List[dict] = carried.to_dict("records") if not carried.empty else []
        pending = detections.loc[~detections["image_id"].isin(done)] if done else detections

        processed = filtered = produced = 0
        items: List[str] = []
        frame_cache_path = None
        frame_cache_image = None

        for record in pending.to_dict("records"):
            frame_path = str(record["frame_path"])
            if frame_cache_path != frame_path:
                try:
                    frame_cache_image = imread(ctx.rel_abs(frame_path))
                except StageError as exc:
                    ctx.warning(f"skipping {record['image_id']}: {exc}")
                    filtered += 1
                    continue
                frame_cache_path = frame_path
            image = frame_cache_image
            assert image is not None  # for type checkers

            bbox = (
                int(record["bbox_x1"]),
                int(record["bbox_y1"]),
                int(record["bbox_x2"]),
                int(record["bbox_y2"]),
            )
            try:
                crop, rect = crop_region(
                    image,
                    bbox,
                    expand_ratio=settings.expand_ratio,
                    square=settings.square,
                    pad_mode=settings.pad_mode,
                    fill=settings.background_fill,
                )
            except StageError as exc:
                ctx.debug(f"skipping {record['image_id']}: {exc}")
                filtered += 1
                continue

            if crop.shape[0] < settings.min_output_side or crop.shape[1] < settings.min_output_side:
                ctx.debug(f"skipping {record['image_id']}: crop smaller than min_output_side")
                filtered += 1
                continue

            if not settings.keep_background:
                mask = saliency_distance(
                    crop, blur_ksize=5
                ) >= float(settings.background_min_distance)
                crop = apply_background_mask(crop, mask, settings.background_fill)

            if settings.output_size:
                crop = cv2.resize(
                    crop,
                    (int(settings.output_size), int(settings.output_size)),
                    interpolation=cv2.INTER_AREA,
                )

            image_id = str(record["image_id"])
            target = ctx.crop_path(str(record["video_id"]), int(record["frame_idx"]), int(record["bbox_idx"]))
            if not ctx.dry_run:
                imwrite(target, crop, quality=settings.jpeg_quality)
            digest = perception_hash(crop, hash_size=cfg.dedup.phash_size)

            rows.append(
                {
                    "image_id": image_id,
                    "video_path": record["video_path"],
                    "video_id": record["video_id"],
                    "frame_idx": int(record["frame_idx"]),
                    "timestamp_ms": int(record.get("timestamp_ms", 0)),
                    "strategy": record.get("strategy", ""),
                    "frame_id": record.get("frame_id", ""),
                    "frame_path": frame_path,
                    "bbox_idx": int(record["bbox_idx"]),
                    "bbox_x1": int(rect[0]),
                    "bbox_y1": int(rect[1]),
                    "bbox_x2": int(rect[2]),
                    "bbox_y2": int(rect[3]),
                    "det_conf": float(record.get("det_conf", 0.0)),
                    "class_name": record.get("class_name", ""),
                    "crop_path": ctx.rel(target),
                    "phash": digest,
                    "quality_score": 0.0,
                    "width": int(crop.shape[1]),
                    "height": int(crop.shape[0]),
                }
            )
            processed += 1
            produced += 1
            items.append(image_id)
            ctx.mark_processed(image_id)

        if not rows:
            raise StageError(
                "no crop was written - check detect.filter (boxes may be too small) "
                "or crop.min_output_side"
            )

        frame = pd.DataFrame(rows)
        if not ctx.dry_run:
            ctx.save("crops", frame)
        ctx.info(f"crops.csv: {len(frame)} crop(s), {len(pending) - produced} skipped")
        return StageOutcome(
            processed=processed,
            filtered=filtered,
            produced=produced,
            items=items,
            outputs=[str(ctx.table("crops"))],
            extra={"expand_ratio": settings.expand_ratio, "square": settings.square},
        )
