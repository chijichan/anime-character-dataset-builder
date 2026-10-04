"""Stage 9 - ``export``: build the ImageFolder dataset and the two manifests.

Layout produced::

    data/dataset/train/{anime}__{character}/{anime}__{character}__{video_id}_f{frame:06d}_b{bbox:02d}.jpg
    data/dataset/val/...
    data/dataset/test/...
    data/metadata/metadata.csv      <- the 16 contract columns, in order
    data/metadata/classes.json      <- {"classes": [{"id": 0, "anime": ..., "character": ...}, ...]}
    data/metadata/class_stats.csv   <- per class x split sample counts

Guarantees enforced here (fail loudly instead of shipping a broken dataset):
every ``image_id`` is unique, no ``anime``/``character``/``split`` value is
empty, and every ``split`` is one of ``train``/``val``/``test``
(:func:`acdb.io.tables.validate_metadata`).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from acdb.errors import StageError
from acdb.io import (
    UNKNOWN_KEY,
    build_classes,
    class_dirname,
    write_classes_json,
    write_metadata_csv,
    write_table,
)
from acdb.pipeline.context import StageContext, StageOutcome
from acdb.pipeline.stage import Stage
from acdb.utils.logging import get_logger
from acdb.utils.naming import final_filename

__all__ = ["ExportStage"]


class ExportStage(Stage):
    name = "export"
    section = "export"
    extra_sections = ("paths",)
    description = "write the ImageFolder dataset, metadata.csv and classes.json"

    def execute(self, ctx: StageContext) -> StageOutcome:
        cfg = ctx.cfg
        settings = cfg.export
        logger = get_logger(self.name)

        frame = ctx.load("split", required=["image_id", "anime", "character", "split", "crop_path"])
        if frame.empty:
            raise StageError("split.csv is empty - run `acdb split` first")

        processed = len(frame)
        dropped_unknown = 0
        if settings.drop_unknown:
            mask = (frame["anime"].astype(str) == UNKNOWN_KEY) | (
                frame["character"].astype(str) == UNKNOWN_KEY
            )
            dropped_unknown = int(mask.sum())
            if dropped_unknown:
                ctx.info(
                    f"dropping {dropped_unknown} image(s) in unknown/ "
                    f"(export.drop_unknown=true)"
                )
            frame = frame.loc[~mask].copy()

        frame["class_key"] = [
            class_dirname(anime, character)
            for anime, character in zip(frame["anime"].astype(str), frame["character"].astype(str))
        ]
        sizes = frame["class_key"].value_counts()
        too_small = [key for key, count in sizes.items() if count < settings.min_images_per_class]
        if too_small:
            ctx.warning(
                f"dropping {len(too_small)} class(es) with fewer than "
                f"{settings.min_images_per_class} image(s): "
                + ", ".join(f"{key}({sizes[key]})" for key in sorted(too_small))
            )
            frame = frame.loc[~frame["class_key"].isin(too_small)].copy()
        if frame.empty:
            raise StageError(
                "nothing left to export. Most likely every cluster is still "
                "unlabelled - fill in data/clusters/cluster_labels.yaml and re-run "
                "`acdb apply-labels` then `acdb split`."
            )

        frame = frame.reset_index(drop=True)
        frame["file_path"] = ""

        if settings.overwrite and not ctx.dry_run:
            self._clean_dataset(ctx)

        copied = linked = skipped = missing = 0
        for position, record in enumerate(frame.to_dict("records")):
            source = ctx.rel_abs(str(record["crop_path"]))
            if not source.is_file():
                missing += 1
                logger.warning(f"source crop is missing, skipping: {record['crop_path']}")
                continue
            name = final_filename(
                str(record["anime"]),
                str(record["character"]),
                str(record["video_id"]),
                int(record["frame_idx"]),
                int(record["bbox_idx"]),
                cfg.naming.final_template,
            )
            target = cfg.paths.dataset_dir / str(record["split"]) / str(record["class_key"]) / name
            frame.at[position, "file_path"] = ctx.rel(target)
            if ctx.dry_run:
                continue
            if target.exists() and not settings.overwrite:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            outcome = self._place(source, target, settings.copy_mode)
            if outcome == "link":
                linked += 1
            else:
                copied += 1

        if missing:
            frame = frame.loc[frame["file_path"] != ""].reset_index(drop=True)
        if frame.empty:
            raise StageError("no image could be exported (every source crop is missing)")

        classes = build_classes(frame["class_key"].tolist())
        stats = self._stats(frame)

        if not ctx.dry_run:
            write_metadata_csv(frame, ctx.table("metadata"))
            write_classes_json(classes, ctx.table("classes"))
            write_table(stats, cfg.export.stats_path, columns=list(stats.columns))

        ctx.info(f"dataset: {len(frame)} image(s) in {len(classes)} class(es)")
        ctx.info("samples per class (train/val/test):")
        for row in stats.to_dict("records"):
            ctx.info(
                f"  {row['class']}: {row['total']} "
                f"(train={row['train']}, val={row['val']}, test={row['test']})"
            )
        ctx.info(
            f"metadata.csv -> {ctx.rel(ctx.table('metadata'))}, "
            f"classes.json -> {ctx.rel(ctx.table('classes'))}"
        )
        return StageOutcome(
            processed=processed,
            filtered=processed - len(frame),
            produced=len(frame),
            items=frame["image_id"].astype(str).tolist(),
            outputs=[
                str(ctx.table("metadata")),
                str(ctx.table("classes")),
                str(cfg.paths.dataset_dir),
            ],
            extra={
                "classes": len(classes),
                "copied": copied,
                "linked": linked,
                "skipped_existing": skipped,
                "dropped_unknown": dropped_unknown,
                "copy_mode": settings.copy_mode,
            },
        )

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _clean_dataset(ctx: StageContext) -> None:
        """Remove previously exported split directories (overwrite mode)."""
        for split_name in ("train", "val", "test"):
            directory = ctx.cfg.paths.dataset_dir / split_name
            if directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def _place(source: Path, target: Path, mode: str) -> str:
        """Copy/link one image; falls back to copying when linking fails."""
        if mode == "hardlink":
            try:
                os.link(source, target)
                return "link"
            except OSError:
                shutil.copy2(source, target)
                return "copy"
        if mode == "symlink":
            try:
                os.symlink(source, target)
                return "link"
            except OSError:
                shutil.copy2(source, target)
                return "copy"
        shutil.copy2(source, target)
        return "copy"

    @staticmethod
    def _stats(frame: pd.DataFrame) -> pd.DataFrame:
        """Per class x split sample counts, for the acceptance report."""
        table = (
            frame.groupby(["class_key", "split"]).size().unstack(fill_value=0)
        )
        for split_name in ("train", "val", "test"):
            if split_name not in table.columns:
                table[split_name] = 0
        table = table.loc[:, ["train", "val", "test"]]
        table["total"] = table.sum(axis=1)
        table = table.reset_index().rename(columns={"class_key": "class"})
        return table.sort_values("total", ascending=False).reset_index(drop=True)
