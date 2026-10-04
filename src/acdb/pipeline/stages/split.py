"""Stage 8 - ``split``: leak-free, class-stratified train/val/test assignment.

Two properties matter more than hitting the ratios exactly:

**No leakage.** By default the unit of assignment is ``(class, video_id)``, so
every crop of one character *from one video* lands in exactly one split. A
character appearing in two videos can legitimately appear in two splits; a
single video's frames never straddle the boundary, which is what would inflate
validation scores.

**Per-class coverage.** Units are handed out greedily to whichever split is
furthest below its target, so a class's frames are distributed in the requested
proportion even when the class only has a handful of units. Classes with fewer
than ``split.min_images_per_class`` images are dropped (and reported).

Output: ``split.csv``. The allocation itself lives in
:func:`allocate_splits`, a pure function with unit tests.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from acdb.errors import StageError
from acdb.io import UNKNOWN_KEY, class_dirname
from acdb.pipeline.context import StageContext, StageOutcome
from acdb.pipeline.stage import Stage

__all__ = ["SplitStage", "allocate_splits", "build_units", "SPLIT_NAMES"]

SPLIT_NAMES: Sequence[str] = ("train", "val", "test")


def allocate_splits(
    unit_sizes: Sequence[int],
    ratios: Mapping[str, float],
    rng: random.Random,
) -> List[str]:
    """Assign each unit to a split, greedily filling the largest deficit first.

    ``unit_sizes`` is the number of images per unit (units are never broken
    apart). Splits with a zero ratio never receive anything unless every ratio
    but one is zero. Units with ``>= 3`` units in their class get at least one
    unit in every non-empty smaller split, so ``val``/``test`` are never empty
    for classes that can afford them.
    """
    active = [name for name in SPLIT_NAMES if ratios.get(name, 0.0) > 0]
    if not active:
        raise StageError("split.ratios leaves no non-empty split")
    if not unit_sizes:
        return []

    total = float(sum(unit_sizes))
    targets = {name: float(ratios.get(name, 0.0)) * total for name in active}
    assigned = {name: 0.0 for name in active}
    counts = {name: 0 for name in active}

    order = list(range(len(unit_sizes)))
    rng.shuffle(order)

    result = [""] * len(unit_sizes)
    for position in order:
        # Primary key: absolute deficit in pixels; secondary: relative deficit.
        choice = max(
            active,
            key=lambda name: (
                targets[name] - assigned[name],
                (targets[name] - assigned[name]) / max(targets[name], 1.0),
                -SPLIT_NAMES.index(name),
            ),
        )
        result[position] = choice
        assigned[choice] += unit_sizes[position]
        counts[choice] += 1

    # Guarantee at least one unit in val/test when the class can afford it.
    if len(unit_sizes) >= 3:
        for name in active:
            if name == "train" or counts[name] > 0:
                continue
            donors = [other for other in active if counts[other] > 1]
            if not donors:
                continue
            donor = max(donors, key=lambda other: counts[other])
            movable = [i for i, value in enumerate(result) if value == donor]
            picked = movable[-1]
            result[picked] = name
            counts[donor] -= 1
            counts[name] += 1
    return result


def build_units(
    frame: pd.DataFrame, *, prevent_leakage: bool
) -> List[Tuple[str, List[int]]]:
    """Group row indices into non-splittable units, per class."""
    units: List[Tuple[str, List[int]]] = []
    keys = list(zip(frame["anime"].astype(str), frame["character"].astype(str)))
    videos = frame["video_id"].astype(str).tolist() if "video_id" in frame.columns else [""] * len(frame)
    if not prevent_leakage:
        for index, key in enumerate(keys):
            units.append((class_dirname(*key), [index]))
        return units

    buckets: Dict[Tuple[str, str], List[int]] = {}
    for index, (key, video) in enumerate(zip(keys, videos)):
        buckets.setdefault((class_dirname(*key), video), []).append(index)
    for (class_key, _video), members in buckets.items():
        units.append((class_key, members))
    return units


class SplitStage(Stage):
    name = "split"
    section = "split"
    description = "class-stratified, leak-free train/val/test split"

    def execute(self, ctx: StageContext) -> StageOutcome:
        cfg = ctx.cfg
        settings = cfg.split
        labeled = ctx.load("labeled", required=["image_id", "anime", "character"])
        if labeled.empty:
            raise StageError("labeled.csv is empty - run `acdb apply-labels` first")

        frame = labeled.copy()
        frame["class_key"] = [
            class_dirname(anime, character)
            for anime, character in zip(frame["anime"].astype(str), frame["character"].astype(str))
        ]

        # Drop classes that are too small to be worth a directory. `unknown` is
        # exempt: it is a review bucket, not a class, and export decides its fate.
        sizes = frame["class_key"].value_counts()
        too_small = [
            key
            for key, count in sizes.items()
            if count < settings.min_images_per_class and not self._is_unknown(key)
        ]
        if too_small:
            ctx.warning(
                f"dropping {len(too_small)} class(es) with fewer than "
                f"{settings.min_images_per_class} image(s): "
                + ", ".join(f"{key}({sizes[key]})" for key in sorted(too_small))
            )
            frame = frame.loc[~frame["class_key"].isin(too_small)].copy()
        if frame.empty:
            raise StageError(
                "every class was dropped by split.min_images_per_class; "
                "lower it or produce more crops"
            )

        rng = random.Random(settings.seed)
        frame = frame.reset_index(drop=True)
        frame["split"] = ""
        active_splits = [name for name in SPLIT_NAMES if settings.ratios.as_dict().get(name, 0.0) > 0]
        relaxed_classes: List[str] = []

        for class_key, group in frame.groupby("class_key", sort=True):
            positions = group.index.tolist()
            sub = frame.loc[positions]
            units = build_units(sub, prevent_leakage=settings.prevent_leakage)
            if settings.prevent_leakage and len(units) < len(active_splits):
                if settings.leakage_fallback == "image":
                    # One video per class cannot fill every split without either
                    # empty val/test or crossing video boundaries. Relaxing to
                    # per-image units for this class keeps the dataset usable;
                    # the warning makes the (small) leakage explicit.
                    relaxed_classes.append(class_key)
                    units = build_units(sub, prevent_leakage=False)
                else:
                    units = [("", [index for index in range(len(sub))])]
            assignment = allocate_splits(
                [len(members) for _key, members in units], settings.ratios.as_dict(), rng
            )
            for (_unit_key, members), split_name in zip(units, assignment):
                for local in members:
                    frame.at[positions[local], "split"] = split_name

        if relaxed_classes:
            ctx.warning(
                f"{len(relaxed_classes)} class(es) appear in fewer videos than there are "
                f"non-empty splits, so they fall back to per-image assignment "
                f"(split.leakage_fallback=image): " + ", ".join(sorted(relaxed_classes))
            )

        if (frame["split"] == "").any():  # pragma: no cover - defensive
            raise StageError("internal error: some rows were not assigned to a split")

        kept = int(len(frame))
        order = {name: index for index, name in enumerate(SPLIT_NAMES)}
        frame = frame.sort_values(
            by=["split", "class_key", "video_id", "frame_idx", "bbox_idx"],
            key=lambda column: column.map(order) if column.name == "split" else column,
            kind="stable",
        ).reset_index(drop=True)

        if not ctx.dry_run:
            ctx.save("split", frame)

        report = frame.groupby(["class_key", "split"]).size().unstack(fill_value=0)
        for split_name in SPLIT_NAMES:
            if split_name not in report.columns:
                report[split_name] = 0
        report = report.loc[:, list(SPLIT_NAMES)]
        ctx.info("split allocation (class x split):")
        for class_key, row in report.iterrows():
            ctx.info(
                f"  {class_key}: train={row['train']} val={row['val']} test={row['test']}"
            )

        totals = {name: int((frame["split"] == name).sum()) for name in SPLIT_NAMES}
        ctx.info(
            f"split.csv: {kept} image(s) -> "
            + ", ".join(f"{name}={count}" for name, count in totals.items())
        )
        return StageOutcome(
            processed=int(len(labeled)),
            filtered=int(len(labeled) - kept),
            produced=kept,
            items=frame["image_id"].astype(str).tolist(),
            outputs=[str(ctx.table("split"))],
            extra={
                "classes": int(frame["class_key"].nunique()),
                **{f"{name}": count for name, count in totals.items()},
                "leak_free": settings.prevent_leakage,
            },
        )

    @staticmethod
    def _is_unknown(class_key: str) -> bool:
        return class_key == f"{UNKNOWN_KEY}__{UNKNOWN_KEY}"
