"""Naming contract: slugs, video ids and the three file-name templates."""

from __future__ import annotations

from pathlib import Path

import pytest

from acdb.io import build_image_id
from acdb.utils.naming import (
    class_dirname,
    crop_filename,
    final_filename,
    frame_filename,
    parse_final_filename,
    slugify,
    video_id_for,
)

FRAME_TEMPLATE = "{video_id}_f{frame_idx:06d}.jpg"
CROP_TEMPLATE = "{video_id}_f{frame_idx:06d}_b{bbox_idx:02d}.jpg"
FINAL_TEMPLATE = "{anime}__{character}__{video_id}_f{frame_idx:06d}_b{bbox_idx:02d}.jpg"


def test_slugify_is_filesystem_safe() -> None:
    assert slugify("Episode 01 - The Beginning!") == "episode_01_the_beginning"
    assert slugify("__leading__trailing__") == "leading_trailing"
    # Runs of separators collapse to a single underscore, so a slug can never
    # contain the "__" that separates anime from character.
    assert "__" not in slugify("a -- b __ c")
    assert slugify("日本語") == "video"
    assert slugify("x" * 100, maxlen=10) == "x" * 10
    assert slugify("con") == "con_v"  # Windows reserved device name


def test_video_id_is_stable_and_path_sensitive() -> None:
    root = Path("data/input")
    first = video_id_for(root / "show_a" / "ep01.mp4", root)
    again = video_id_for(root / "show_a" / "ep01.mp4", root)
    other_folder = video_id_for(root / "show_b" / "ep01.mp4", root)

    assert first == again, "video_id must be deterministic"
    assert first != other_folder, "same stem in another folder must differ"
    assert first.startswith("ep01_") and len(first) == len("ep01_") + 8
    assert "__" not in first


def test_video_id_is_portable_across_roots() -> None:
    """Moving the whole input tree must not change ids."""
    assert video_id_for(Path("a/b/c/ep07.mkv"), Path("a")) == video_id_for(
        Path("x/b/c/ep07.mkv"), Path("x")
    )


def test_frame_and_crop_names_follow_the_contract() -> None:
    assert frame_filename("ep01_1a2b3c4d", 42, FRAME_TEMPLATE) == "ep01_1a2b3c4d_f000042.jpg"
    assert (
        crop_filename("ep01_1a2b3c4d", 42, 3, CROP_TEMPLATE) == "ep01_1a2b3c4d_f000042_b03.jpg"
    )
    assert build_image_id("ep01_1a2b3c4d", 42, 3) == "ep01_1a2b3c4d_f000042_b03"


def test_final_name_and_round_trip() -> None:
    name = final_filename("show_a", "hikari", "ep01_1a2b3c4d", 7, 2, FINAL_TEMPLATE)
    assert name == "show_a__hikari__ep01_1a2b3c4d_f000007_b02.jpg"
    assert parse_final_filename(name) == {
        "anime": "show_a",
        "character": "hikari",
        "video_id": "ep01_1a2b3c4d",
        "frame_idx": 7,
        "bbox_idx": 2,
    }
    assert class_dirname("show_a", "hikari") == "show_a__hikari"


@pytest.mark.parametrize("bad", ["nope.jpg", "a__b.jpg", "a__b__c.jpg", "a__b__c_f1_b2.png"])
def test_parse_final_filename_rejects_malformed(bad: str) -> None:
    assert parse_final_filename(bad) is None


def test_bbox_index_is_zero_padded() -> None:
    """bbox_idx must stay two digits - documented limit is 99 boxes per frame."""
    assert crop_filename("v_00000000", 0, 9, CROP_TEMPLATE).endswith("_b09.jpg")
    assert crop_filename("v_00000000", 0, 99, CROP_TEMPLATE).endswith("_b99.jpg")
