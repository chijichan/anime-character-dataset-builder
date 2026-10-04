"""Stage 5 - ``filter``: quality gating and the composite ``quality_score``.

Measures every surviving crop once (Laplacian variance, clipping ratios,
contrast, brightness, Hasler-Susstrunk colorfulness), turns them into the
composite score described in :mod:`acdb.pipeline.scoring`, and applies the hard
gates from ``quality``. Rejects are optionally moved to ``data/rejected/`` so a
human can check whether a threshold is too aggressive.

Output: ``quality.csv`` - the rows dedup kept, plus the metrics, the score and
the quality verdict in ``kept``.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from acdb.errors import StageError
from acdb.pipeline.context import StageContext, StageOutcome
from acdb.pipeline.scoring import METRIC_COLUMNS, evaluate
from acdb.pipeline.stage import Stage
from acdb.utils.image import imread

__all__ = ["FilterStage"]


class FilterStage(Stage):
    name = "filter"
    section = "quality"
    description = "score crops for blur/exposure/contrast and drop the bad ones"

    def execute(self, ctx: StageContext) -> StageOutcome:
        settings = ctx.cfg.quality
        dedup = ctx.load("dedup", required=["image_id", "crop_path", "kept"])
        candidates = dedup.loc[dedup["kept"].astype(bool)].copy()
        if candidates.empty:
            raise StageError(
                "dedup.csv has no surviving crop - every image was a duplicate. "
                "Loosen dedup.phash_hamming / dedup.cosine_threshold."
            )

        frame = candidates.copy()
        scores: List[float] = []
        metrics_rows: List[Dict[str, float]] = []
        verdicts: List[bool] = []
        reasons: Dict[str, int] = {}
        processed = rejected = 0

        for record in candidates.to_dict("records"):
            image_id = str(record["image_id"])
            try:
                image = imread(ctx.rel_abs(str(record["crop_path"])))
            except StageError as exc:
                ctx.warning(f"skipping {image_id}: {exc}")
                scores.append(0.0)
                metrics_rows.append({column: 0.0 for column in METRIC_COLUMNS})
                verdicts.append(False)
                reasons["unreadable"] = reasons.get("unreadable", 0) + 1
                rejected += 1
                continue

            verdict = evaluate(image, settings)
            scores.append(verdict.score)
            metrics_rows.append(verdict.metrics)
            verdicts.append(verdict.passed)
            processed += 1
            if not verdict.passed:
                rejected += 1
                for reason in verdict.reasons:
                    reasons[reason] = reasons.get(reason, 0) + 1

        frame["quality_score"] = [round(value, 6) for value in scores]
        for column in METRIC_COLUMNS:
            frame[column] = [round(float(row.get(column, 0.0)), 6) for row in metrics_rows]
        frame["kept"] = verdicts

        if settings.move_rejected and not ctx.dry_run:
            self._move_rejected(ctx, frame)

        kept = int(frame["kept"].sum())
        if not ctx.dry_run:
            ctx.save("quality", frame)
        ctx.info(
            f"quality.csv: {len(frame)} crop(s), {kept} kept, {rejected} rejected"
            + (f" ({reasons})" if reasons else "")
        )
        if kept == 0:
            raise StageError(
                "every crop failed the quality gates. Lower quality.min_score / "
                "min_laplacian_var, then re-run `acdb filter`."
            )
        extra = {"score_mean": round(float(frame.loc[frame["kept"], "quality_score"].mean()), 4)}
        if reasons:
            extra["rejected"] = ",".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
        return StageOutcome(
            processed=processed,
            filtered=rejected,
            produced=kept,
            items=frame["image_id"].astype(str).tolist(),
            outputs=[str(ctx.table("quality"))],
            extra=extra,
        )

    # -- helpers ----------------------------------------------------------- #
    def _move_rejected(self, ctx: StageContext, frame: pd.DataFrame) -> None:
        """Move rejected crops to ``rejected_dir`` and update their paths."""
        from shutil import move

        for position in frame.index[~frame["kept"].astype(bool)]:
            source = ctx.rel_abs(str(frame.at[position, "crop_path"]))
            if not source.is_file():
                continue
            destination = ctx.cfg.paths.rejected_dir / str(frame.at[position, "video_id"]) / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                move(str(source), str(destination))
                frame.at[position, "crop_path"] = ctx.rel(destination)
            except OSError as exc:  # pragma: no cover - filesystem dependent
                ctx.warning(f"could not move rejected crop {source.name}: {exc}")
