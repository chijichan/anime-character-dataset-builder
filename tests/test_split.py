"""Split logic: ratio targeting, per-class coverage, leakage-free grouping."""

from __future__ import annotations

import random

import pandas as pd
import pytest

from acdb.pipeline.stages.split import SPLIT_NAMES, allocate_splits, build_units

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _counts(assignment, sizes=None):
    sizes = sizes or [1] * len(assignment)
    totals = {name: 0 for name in SPLIT_NAMES}
    for name, size in zip(assignment, sizes):
        totals[name] += size
    return totals


def test_allocate_splits_hits_the_requested_ratios() -> None:
    sizes = [10] * 10  # 100 images across 10 units
    assignment = allocate_splits(sizes, RATIOS, random.Random(0))
    totals = _counts(assignment, sizes)
    assert totals["train"] == 80
    assert totals["val"] == 10
    assert totals["test"] == 10


def test_allocate_splits_is_deterministic_for_a_seed() -> None:
    sizes = [3, 1, 4, 1, 5, 9, 2, 6]
    first = allocate_splits(sizes, RATIOS, random.Random(7))
    second = allocate_splits(sizes, RATIOS, random.Random(7))
    third = allocate_splits(sizes, RATIOS, random.Random(8))
    assert first == second
    assert first != third, "a different seed must produce a different assignment"


def test_allocate_splits_never_breaks_a_unit() -> None:
    sizes = [5, 5, 5, 5]
    assignment = allocate_splits(sizes, RATIOS, random.Random(1))
    assert len(assignment) == len(sizes), "one label per unit, units stay whole"


def test_small_classes_still_reach_every_split() -> None:
    """Three units must give val and test at least one unit each."""
    assignment = allocate_splits([1, 1, 1], RATIOS, random.Random(3))
    assert set(assignment) == {"train", "val", "test"}


def test_zero_ratio_splits_receive_nothing() -> None:
    assignment = allocate_splits([1] * 12, {"train": 1.0, "val": 0.0, "test": 0.0}, random.Random(0))
    assert set(assignment) == {"train"}


def test_two_way_split() -> None:
    assignment = allocate_splits([1] * 10, {"train": 0.5, "val": 0.5, "test": 0.0}, random.Random(0))
    assert set(assignment) == {"train", "val"}


def test_allocate_splits_rejects_empty_ratios() -> None:
    from acdb.errors import StageError

    with pytest.raises(StageError):
        allocate_splits([1, 2], {"train": 0.0, "val": 0.0, "test": 0.0}, random.Random(0))


def test_allocate_splits_handles_no_units() -> None:
    assert allocate_splits([], RATIOS, random.Random(0)) == []


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_build_units_groups_by_class_and_video_when_leak_free() -> None:
    frame = _frame(
        [
            {"anime": "a", "character": "x", "video_id": "v1"},
            {"anime": "a", "character": "x", "video_id": "v1"},
            {"anime": "a", "character": "x", "video_id": "v2"},
            {"anime": "a", "character": "y", "video_id": "v1"},
        ]
    )
    units = build_units(frame, prevent_leakage=True)
    # (a__x, v1) -> rows 0,1 | (a__x, v2) -> row 2 | (a__y, v1) -> row 3
    assert sorted(sorted(members) for _key, members in units) == [[0, 1], [2], [3]]


def test_build_units_per_image_when_leakage_is_allowed() -> None:
    frame = _frame(
        [
            {"anime": "a", "character": "x", "video_id": "v1"},
            {"anime": "a", "character": "x", "video_id": "v1"},
        ]
    )
    units = build_units(frame, prevent_leakage=False)
    assert sorted(len(members) for _key, members in units) == [1, 1]


def test_units_from_one_video_never_cross_splits() -> None:
    """The core anti-leakage property, checked end to end on one class."""
    frame = _frame(
        [
            {"anime": "a", "character": "x", "video_id": "v1"},
            {"anime": "a", "character": "x", "video_id": "v1"},
            {"anime": "a", "character": "x", "video_id": "v1"},
            {"anime": "a", "character": "x", "video_id": "v2"},
            {"anime": "a", "character": "x", "video_id": "v2"},
            {"anime": "a", "character": "x", "video_id": "v2"},
        ]
    )
    units = build_units(frame, prevent_leakage=True)
    assignment = allocate_splits(
        [len(members) for _key, members in units], RATIOS, random.Random(5)
    )
    per_video = {}
    for (_key, members), split_name in zip(units, assignment):
        for member in members:
            per_video.setdefault(frame.at[member, "video_id"], set()).add(split_name)
    assert all(len(splits) == 1 for splits in per_video.values())
