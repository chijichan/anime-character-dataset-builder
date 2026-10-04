"""Metadata contract: columns, validation, round-trip and classes.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from acdb.errors import MetadataError
from acdb.io import (
    METADATA_COLUMNS,
    build_classes,
    read_classes_json,
    read_table,
    validate_metadata,
    write_classes_json,
    write_metadata_csv,
    write_table,
)
from acdb.io.labels import UNKNOWN_KEY, read_labels, write_label_template


def _row(**overrides) -> dict:
    row = {
        "image_id": "v1_f000001_b00",
        "anime": "show",
        "character": "hero",
        "split": "train",
        "video_path": "data/input/v1.mp4",
        "video_id": "v1",
        "frame_idx": 1,
        "timestamp_ms": 100,
        "bbox_x1": 0,
        "bbox_y1": 0,
        "bbox_x2": 32,
        "bbox_y2": 32,
        "det_conf": 0.9,
        "cluster_id": 0,
        "quality_score": 0.5,
        "file_path": "data/dataset/train/show__hero/x.jpg",
    }
    row.update(overrides)
    return row


def test_metadata_columns_are_the_contract() -> None:
    assert list(METADATA_COLUMNS) == [
        "image_id",
        "anime",
        "character",
        "split",
        "video_path",
        "video_id",
        "frame_idx",
        "timestamp_ms",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "det_conf",
        "cluster_id",
        "quality_score",
        "file_path",
    ]


def test_write_metadata_projects_and_orders_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame([_row(), _row(image_id="v1_f000002_b00", frame_idx=2)])
    frame["crop_path"] = "extra/column.jpg"  # must be dropped from metadata.csv
    target = tmp_path / "metadata.csv"
    write_metadata_csv(frame, target)

    written = pd.read_csv(target, dtype=str, keep_default_na=False)
    assert list(written.columns) == list(METADATA_COLUMNS)
    assert "crop_path" not in written.columns
    assert written["image_id"].tolist() == ["v1_f000001_b00", "v1_f000002_b00"]
    assert target.read_bytes()[:3] != b"\xef\xbb\xbf", "metadata.csv must be UTF-8 without BOM"


def test_metadata_round_trip_keeps_dtypes(tmp_path: Path) -> None:
    frame = pd.DataFrame([_row()])
    target = tmp_path / "metadata.csv"
    write_metadata_csv(frame, target)
    loaded = read_table(target, table="metadata")
    assert loaded["frame_idx"].dtype.kind == "i"
    assert loaded["det_conf"].dtype.kind == "f"
    assert loaded["anime"].iloc[0] == "show"
    assert int(loaded["timestamp_ms"].iloc[0]) == 100


def test_validate_metadata_rejects_bad_tables() -> None:
    ok = pd.DataFrame([_row()], columns=list(METADATA_COLUMNS))
    validate_metadata(ok)

    with pytest.raises(MetadataError, match="column mismatch"):
        validate_metadata(ok.loc[:, [c for c in METADATA_COLUMNS if c != "split"]])

    with pytest.raises(MetadataError, match="duplicate image_id"):
        validate_metadata(pd.DataFrame([_row(), _row()], columns=list(METADATA_COLUMNS)))

    with pytest.raises(MetadataError, match="empty 'character'"):
        validate_metadata(
            pd.DataFrame([_row(character="")], columns=list(METADATA_COLUMNS))
        )

    with pytest.raises(MetadataError, match="unknown split"):
        validate_metadata(pd.DataFrame([_row(split="valid")], columns=list(METADATA_COLUMNS)))


def test_write_table_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "table.csv"
    write_table(pd.DataFrame([{"a": 1, "b": "x"}]), target)
    assert target.is_file()
    assert list(target.parent.glob("*.tmp")) == []
    assert list(target.parent.glob(".*.tmp")) == []


def test_read_table_reports_missing_columns(tmp_path: Path) -> None:
    target = tmp_path / "broken.csv"
    pd.DataFrame([{"image_id": "a"}]).to_csv(target, index=False)
    with pytest.raises(MetadataError, match="missing required column"):
        read_table(target, required=["image_id", "crop_path"])


def test_classes_json_is_sorted_and_numbered(tmp_path: Path) -> None:
    entries = build_classes(["show_b__ren", "show_a__hikari", "show_a__kaito"])
    assert [entry.id for entry in entries] == [0, 1, 2]
    assert [entry.key for entry in entries] == ["show_a__hikari", "show_a__kaito", "show_b__ren"]

    target = tmp_path / "classes.json"
    write_classes_json(entries, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["classes"][0] == {"id": 0, "anime": "show_a", "character": "hikari"}
    assert read_classes_json(target) == entries


def test_read_classes_json_validates_shape(tmp_path: Path) -> None:
    target = tmp_path / "classes.json"
    target.write_text('{"nope": []}', encoding="utf-8")
    with pytest.raises(MetadataError):
        read_classes_json(target)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def test_label_template_then_yaml_round_trip(tmp_path: Path) -> None:
    clusters_dir = tmp_path / "clusters"
    clusters_dir.mkdir()
    (clusters_dir / "cluster_000").mkdir()
    (clusters_dir / "cluster_001").mkdir()

    write_label_template(
        clusters_dir,
        {0: {"size": 5, "suggested_hint": "show_a", "preview": "data/input/show_a/ep01.mp4"},
         1: {"size": 2}},
    )
    labels = read_labels(clusters_dir)
    assert labels[0].is_unknown and labels[1].is_unknown

    (clusters_dir / "cluster_labels.yaml").write_text(
        "clusters:\n  0: show_a__hikari\n  1: {anime: show_a, character: kaito}\n",
        encoding="utf-8",
    )
    labels = read_labels(clusters_dir)
    assert labels[0].key == "show_a__hikari"
    assert labels[0].source == "cluster_labels.yaml"
    assert labels[1].key == "show_a__kaito"


def test_label_template_is_not_clobbered(tmp_path: Path) -> None:
    clusters_dir = tmp_path / "clusters"
    clusters_dir.mkdir()
    target = clusters_dir / "cluster_labels.yaml"
    target.write_text("clusters:\n  0: show_a__hikari\n", encoding="utf-8")
    write_label_template(clusters_dir, {0: {"size": 1}}, overwrite=False)
    assert "show_a__hikari" in target.read_text(encoding="utf-8")
    write_label_template(clusters_dir, {0: {"size": 1}}, overwrite=True)
    assert "show_a__hikari" not in target.read_text(encoding="utf-8")


def test_renamed_folder_wins_when_yaml_says_unknown(tmp_path: Path) -> None:
    clusters_dir = tmp_path / "clusters"
    (clusters_dir / "cluster_004__show_b__yuki").mkdir(parents=True)
    (clusters_dir / "cluster_labels.yaml").write_text(
        "clusters:\n  4: unknown\n", encoding="utf-8"
    )
    labels = read_labels(clusters_dir)
    assert labels[4].key == "show_b__yuki"
    assert labels[4].source == "cluster_004__show_b__yuki"


def test_yaml_wins_over_a_conflicting_folder_name(tmp_path: Path) -> None:
    clusters_dir = tmp_path / "clusters"
    (clusters_dir / "cluster_002__show_b__ren").mkdir(parents=True)
    (clusters_dir / "cluster_labels.yaml").write_text(
        "clusters:\n  2: show_a__hikari\n", encoding="utf-8"
    )
    labels = read_labels(clusters_dir)
    assert labels[2].key == "show_a__hikari"


def test_folder_suffix_form_is_supported(tmp_path: Path) -> None:
    clusters_dir = tmp_path / "clusters"
    (clusters_dir / "show_c__mei__cluster_007").mkdir(parents=True)
    labels = read_labels(clusters_dir)
    assert labels[7].key == "show_c__mei"


def test_invalid_label_value_is_rejected(tmp_path: Path) -> None:
    clusters_dir = tmp_path / "clusters"
    clusters_dir.mkdir()
    (clusters_dir / "cluster_labels.yaml").write_text(
        "clusters:\n  0: hikari\n", encoding="utf-8"
    )
    with pytest.raises(MetadataError, match="anime"):
        read_labels(clusters_dir)


def test_unknown_constant_is_used() -> None:
    assert UNKNOWN_KEY == "unknown"
