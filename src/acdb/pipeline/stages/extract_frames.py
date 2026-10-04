"""Stage 1 - ``extract-frames``: decode videos into candidate frames.

Three selection strategies, freely combined (the union is taken, duplicates
merged):

``interval``
    every N seconds (``interval_seconds``) or every N frames
    (``interval_frames``, which wins when set). One sequential decode pass.
``scene``
    shot boundaries via PySceneDetect, or the built-in HSV histogram detector
    when PySceneDetect is not installed.
``keyframe``
    container I-frames via PyAV. Falls back to scene cuts (and says so) when
    PyAV is missing, because "no I-frame information available" is a property of
    the file/decoder, not a reason to fail the run.

Output: ``data/frames/{video_id}/{video_id}_f{frame_idx:06d}.jpg`` plus
``frames.csv``. Videos already listed in the stage state are skipped and their
rows are carried over, so an interrupted run resumes per file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import pandas as pd

from acdb.errors import StageError
from acdb.io import write_table
from acdb.pipeline.context import StageContext, StageOutcome, reusable_items
from acdb.pipeline.stage import Stage
from acdb.utils.image import imwrite, mean_abs_diff, resize_long_side
from acdb.utils.logging import get_logger
from acdb.utils.video import (
    VideoMeta,
    detect_scene_cuts,
    find_videos,
    get_keyframe_indices,
    iter_selected_frames,
    probe_video,
)

__all__ = ["ExtractFramesStage"]


@dataclass
class _Plan:
    meta: VideoMeta
    indices: List[int] = field(default_factory=list)
    strategies: Dict[int, List[str]] = field(default_factory=dict)


class ExtractFramesStage(Stage):
    name = "extract_frames"
    section = "extract_frames"
    extra_sections = ("video",)
    description = "decode videos into frames (interval / scene / keyframe)"

    # -- planning ---------------------------------------------------------- #
    def _build_plan(self, ctx: StageContext, video_path) -> _Plan:
        cfg = ctx.cfg
        settings = cfg.extract_frames
        meta = probe_video(
            video_path,
            input_root=cfg.paths.input_dir,
            slug_maxlen=cfg.naming.video_id_slug_maxlen,
            hash_len=cfg.naming.video_id_hash_len,
        )
        plan = _Plan(meta=meta)
        strategies = [s.lower() for s in settings.strategies]

        def _add(indices: Sequence[int], label: str) -> None:
            for index in indices:
                if index < 0:
                    continue
                if index not in plan.strategies:
                    plan.strategies[index] = []
                    plan.indices.append(index)
                if label not in plan.strategies[index]:
                    plan.strategies[index].append(label)

        if "interval" in strategies:
            stride = settings.interval_frames or max(
                1, int(round(settings.interval_seconds * meta.fps))
            )
            _add(range(0, max(1, meta.frame_count), stride), "interval")

        if "scene" in strategies:
            cuts = detect_scene_cuts(
                video_path,
                threshold=settings.scene_threshold,
                min_gap_seconds=settings.scene_min_gap_seconds,
                fps=meta.fps,
            )
            _add(cuts, "scene")

        if "keyframe" in strategies:
            keyframes = get_keyframe_indices(video_path)
            if keyframes is None:
                get_logger(self.name).warning(
                    f"{meta.video_id}: no I-frame info available; "
                    f"using scene cuts for the keyframe strategy"
                )
                keyframes = detect_scene_cuts(
                    video_path,
                    threshold=settings.scene_threshold,
                    min_gap_seconds=settings.scene_min_gap_seconds,
                    fps=meta.fps,
                )
            _add(keyframes, "keyframe")

        plan.indices.sort()
        if not plan.indices:
            get_logger(self.name).warning(
                f"{meta.video_id}: no frame selected by {strategies}; falling back to frame 0"
            )
            _add([0], "fallback")

        if settings.max_frames_per_video and len(plan.indices) > settings.max_frames_per_video:
            plan.indices = _subsample(plan.indices, int(settings.max_frames_per_video))
        return plan

    # -- execution --------------------------------------------------------- #
    def execute(self, ctx: StageContext) -> StageOutcome:
        cfg = ctx.cfg
        settings = cfg.extract_frames
        # Decoding one video can take minutes, so checkpoint after every video.
        ctx.flush_every = 1
        videos = find_videos(
            cfg.paths.input_dir,
            cfg.video.extensions,
            recursive=cfg.video.recursive,
            skip_hidden=cfg.video.skip_hidden,
            limit=cfg.project.max_videos,
        )
        if not videos:
            raise StageError(
                f"no video found in {cfg.paths.input_dir} with extensions "
                f"{list(cfg.video.extensions)}. Put your footage there (nested "
                f"subdirectories are fine) and re-run."
            )
        ctx.info(f"found {len(videos)} video(s) under {cfg.paths.input_dir}")

        done = reusable_items(
            ctx.resume_items, (ctx.rel(path) for path in videos)
        )
        carried = ctx.carry_over("frames", "video_path", done)

        rows: List[dict] = list(carried.to_dict("records")) if not carried.empty else []
        processed = filtered = produced = 0
        items: List[str] = []
        skipped_videos = 0
        failed_videos = 0

        for video_path in videos:
            relative = ctx.rel(video_path)
            if relative in done:
                skipped_videos += 1
                continue
            try:
                plan = self._build_plan(ctx, video_path)
                video_rows, emitted, dropped = self._extract_video(ctx, plan, video_path)
            except StageError as exc:
                failed_videos += 1
                ctx.warning(f"skipping {relative}: {exc}")
                continue

            rows.extend(video_rows)
            produced += emitted
            filtered += dropped
            processed += 1
            items.append(relative)
            ctx.mark_processed(relative)
            ctx.info(
                f"{plan.meta.video_id}: {plan.meta.duration_sec:.1f}s, "
                f"{len(plan.indices)} candidate frame(s) -> {emitted} written, {dropped} static"
            )

        if not rows:
            raise StageError(
                "no frame could be extracted from any input video (see the log above)"
            )
        if failed_videos and failed_videos == len(videos):
            raise StageError("every input video failed to decode")

        frame = pd.DataFrame(rows)
        if not ctx.dry_run:
            write_table(frame, ctx.table("frames"), columns=None)
        ctx.info(
            f"frames.csv: {len(frame)} row(s) "
            f"({skipped_videos} video(s) resumed from state)"
        )
        return StageOutcome(
            processed=processed,
            filtered=filtered,
            produced=produced,
            items=items,
            outputs=[str(ctx.table("frames"))],
            extra={
                "videos": len(videos),
                "resumed_videos": skipped_videos,
                "failed_videos": failed_videos,
                "strategies": ",".join(settings.strategies),
            },
        )

    # -- per video --------------------------------------------------------- #
    def _extract_video(self, ctx: StageContext, plan: _Plan, video_path) -> tuple:
        settings = ctx.cfg.extract_frames
        video_id = plan.meta.video_id
        rows: List[dict] = []
        emitted = 0
        dropped = 0
        previous = None

        for frame_idx, image in iter_selected_frames(video_path, plan.indices):
            image = resize_long_side(image, settings.resize_long_side)
            if settings.skip_static and previous is not None:
                difference = mean_abs_diff(previous, image) if previous.shape == image.shape else 999.0
                if difference < settings.static_diff_threshold:
                    dropped += 1
                    ctx.debug(f"{video_id} f{frame_idx}: static frame skipped")
                    continue
            previous = image

            target = ctx.frame_path(video_id, frame_idx)
            if not ctx.dry_run:
                imwrite(target, image, quality=settings.jpeg_quality)
            rows.append(
                {
                    "frame_id": f"{video_id}_f{frame_idx:06d}",
                    "video_id": video_id,
                    "video_path": ctx.rel(video_path),
                    "frame_idx": int(frame_idx),
                    "timestamp_ms": plan.meta.timestamp_ms(frame_idx),
                    "strategy": "+".join(plan.strategies.get(frame_idx, ["interval"])),
                    "frame_path": ctx.rel(target),
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }
            )
            emitted += 1
        return rows, emitted, dropped


def _subsample(indices: Sequence[int], limit: int) -> List[int]:
    """Keep ``limit`` evenly spaced indices (always including the first and last)."""
    if limit <= 0 or len(indices) <= limit:
        return list(indices)
    if limit == 1:
        return [indices[0]]
    step = (len(indices) - 1) / (limit - 1)
    picked = {indices[min(len(indices) - 1, int(round(i * step)))] for i in range(limit)}
    return sorted(picked)
