"""Table schemas, atomic CSV IO and the metadata/classes output contract.

Pipeline stages communicate through CSV tables in ``data/metadata/``:

===================  ==================================================================
``frames.csv``       one row per extracted frame
``detections.csv``   one row per accepted detection (raw detector geometry)
``crops.csv``        one row per written crop file
``dedup.csv``        ``crops.csv`` + ``kept`` / ``dedup_group``
``quality.csv``      ``dedup.csv`` + quality metrics and ``quality_score``
``clusters.csv``     ``quality.csv`` + ``cluster_id``
``labeled.csv``      ``clusters.csv`` + ``anime`` / ``character``
``split.csv``        ``labeled.csv`` + ``split``
``metadata.csv``     **the deliverable**: exactly the 16 contract columns, in order
``classes.json``     **the deliverable**: class id <-> (anime, character)
===================  ==================================================================

Every table shares one superset schema, so each stage only adds columns. All
writes go through :func:`write_table`, which stages into a ``*.tmp`` file and
renames, so an interrupted run never leaves a truncated CSV.

``file_path`` / ``video_path`` / ``crop_path`` are stored **relative to the
project root, with POSIX separators**, which keeps the manifest portable across
machines and operating systems.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from acdb.errors import MetadataError
from acdb.utils.atomic import temp_sibling, write_text_atomic

__all__ = [
    "METADATA_COLUMNS",
    "TABLE_FILES",
    "COLUMN_KINDS",
    "IMAGES_COLUMNS",
    "read_table",
    "write_table",
    "write_metadata_csv",
    "validate_metadata",
    "write_classes_json",
    "read_classes_json",
    "build_image_id",
    "relpath",
    "abspath",
    "table_path",
    "TABLE_SCHEMAS",
]

#: The 16 columns of the deliverable ``metadata.csv``, in the required order.
METADATA_COLUMNS: Sequence[str] = (
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
)

#: Column name -> storage kind. Used to coerce dtypes after a CSV round-trip
#: (an all-empty column otherwise comes back as ``object``/``NaN``).
COLUMN_KINDS: Mapping[str, str] = {
    # identity / provenance
    "frame_id": "str",
    "image_id": "str",
    "video_path": "str",
    "video_id": "str",
    "strategy": "str",
    "frame_path": "str",
    "crop_path": "str",
    "class_name": "str",
    "phash": "str",
    "dedup_group": "str",
    "anime": "str",
    "character": "str",
    "split": "str",
    "file_path": "str",
    # geometry / timing
    "frame_idx": "int",
    "timestamp_ms": "int",
    "bbox_idx": "int",
    "bbox_x1": "int",
    "bbox_y1": "int",
    "bbox_x2": "int",
    "bbox_y2": "int",
    "width": "int",
    "height": "int",
    "cluster_id": "int",
    "kept": "bool",
    "det_conf": "float",
    "quality_score": "float",
    "laplacian_var": "float",
    "overexposed_ratio": "float",
    "underexposed_ratio": "float",
    "colorfulness": "float",
    "contrast": "float",
    "brightness": "float",
}

#: Canonical column order of the shared superset schema.
IMAGES_COLUMNS: Sequence[str] = (
    "image_id",
    "video_path",
    "video_id",
    "frame_idx",
    "timestamp_ms",
    "bbox_idx",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "det_conf",
    "class_name",
    "strategy",
    "frame_id",
    "frame_path",
    "crop_path",
    "phash",
    "dedup_group",
    "kept",
    "quality_score",
    "laplacian_var",
    "overexposed_ratio",
    "underexposed_ratio",
    "colorfulness",
    "contrast",
    "brightness",
    "cluster_id",
    "anime",
    "character",
    "split",
    "file_path",
    "width",
    "height",
)

FRAME_COLUMNS: Sequence[str] = (
    "frame_id", "video_id", "video_path", "frame_idx", "timestamp_ms",
    "strategy", "frame_path", "width", "height",
)

#: File name of every intermediate table inside ``metadata_dir``.
TABLE_FILES: Mapping[str, str] = {
    "frames": "frames.csv",
    "detections": "detections.csv",
    "crops": "crops.csv",
    "dedup": "dedup.csv",
    "quality": "quality.csv",
    "clusters": "clusters.csv",
    "labeled": "labeled.csv",
    "split": "split.csv",
    "metadata": "metadata.csv",
    "classes": "classes.json",
    "stats": "class_stats.csv",
    "features": "features.npz",
}

#: Columns each stage guarantees on its output table.
TABLE_SCHEMAS: Mapping[str, Sequence[str]] = {
    "frames": FRAME_COLUMNS,
    "detections": (
        "video_path", "video_id", "frame_idx", "timestamp_ms", "strategy",
        "frame_path", "bbox_idx", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "det_conf", "class_name", "width", "height",
    ),
    "crops": (
        "image_id", "video_path", "video_id", "frame_idx", "timestamp_ms",
        "bbox_idx", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "det_conf",
        "class_name", "crop_path", "phash", "quality_score",
    ),
    "dedup": (
        "image_id", "crop_path", "phash", "kept", "dedup_group", "quality_score",
    ),
    "quality": (
        "image_id", "crop_path", "kept", "quality_score", "laplacian_var",
        "overexposed_ratio", "underexposed_ratio", "colorfulness", "contrast",
        "brightness",
    ),
    "clusters": ("image_id", "crop_path", "cluster_id"),
    "labeled": ("image_id", "crop_path", "cluster_id", "anime", "character"),
    "split": ("image_id", "crop_path", "anime", "character", "split"),
    "metadata": METADATA_COLUMNS,
}


def table_path(metadata_dir: Path, key: str) -> Path:
    """Resolve a logical table name to its file inside ``metadata_dir``."""
    try:
        return Path(metadata_dir) / TABLE_FILES[key]
    except KeyError as exc:
        raise MetadataError(
            f"unknown table {key!r}; known: {sorted(TABLE_FILES)}"
        ) from exc


def relpath(path: Path | str, root: Path) -> str:
    """POSIX-style path relative to ``root`` (falls back to the absolute path)."""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def abspath(path: str, root: Path) -> Path:
    """Inverse of :func:`relpath`."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (Path(root) / candidate)


def build_image_id(video_id: str, frame_idx: int, bbox_idx: int) -> str:
    """Stable primary key of an image: mirrors the crop file stem."""
    return f"{video_id}_f{int(frame_idx):06d}_b{int(bbox_idx):02d}"


# --------------------------------------------------------------------------- #
# Reading / writing
# --------------------------------------------------------------------------- #


def _coerce_column(series: pd.Series, kind: str) -> pd.Series:
    if kind == "int":
        return pd.to_numeric(series, errors="coerce").fillna(-1).astype("int64")
    if kind == "float":
        return pd.to_numeric(series, errors="coerce").astype("float64")
    if kind == "bool":
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(0).astype("int64").astype(bool)
        return (
            series.fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes", "y", "t"})
        )
    return series.fillna("").astype(str)


def read_table(
    path: Path | str,
    *,
    required: Sequence[str] | None = None,
    table: Optional[str] = None,
) -> pd.DataFrame:
    """Read a pipeline CSV, coercing dtypes and validating required columns."""
    file_path = Path(path)
    if not file_path.is_file():
        raise MetadataError(f"table not found: {file_path}")
    try:
        frame = pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding="utf-8")
    except (OSError, pd.errors.ParserError) as exc:
        raise MetadataError(f"cannot read {file_path}: {exc}") from exc

    if table is not None:
        required = TABLE_SCHEMAS.get(table, ()) if required is None else required

    missing = [column for column in (required or ()) if column not in frame.columns]
    if missing:
        raise MetadataError(
            f"{file_path} is missing required column(s) {missing}. "
            f"Re-run the stage that produces this table."
        )
    for column in frame.columns:
        frame[column] = _coerce_column(frame[column], COLUMN_KINDS.get(column, "str"))
    return frame


def ordered_columns(columns: Iterable[str]) -> List[str]:
    """Return ``columns`` sorted into the canonical superset order."""
    known = [c for c in IMAGES_COLUMNS if c in set(columns)]
    extra = [c for c in columns if c not in set(IMAGES_COLUMNS)]
    return known + extra


def write_table(
    frame: pd.DataFrame,
    path: Path | str,
    *,
    columns: Sequence[str] | None = None,
) -> Path:
    """Atomically write ``frame`` as UTF-8 CSV (no BOM, LF endings, no index)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise MetadataError(f"cannot write {target}: missing column(s) {missing}")
        frame = frame.loc[:, list(columns)]

    staging = temp_sibling(target)
    try:
        frame.to_csv(
            staging,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            na_rep="",
        )
        os.replace(staging, target)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return target


def validate_metadata(frame: pd.DataFrame) -> None:
    """Assert the hard output contract of ``metadata.csv``."""
    columns = list(frame.columns)
    if columns != list(METADATA_COLUMNS):
        raise MetadataError(
            "metadata.csv column mismatch.\n"
            f"  expected: {list(METADATA_COLUMNS)}\n"
            f"  actual:   {columns}"
        )
    if frame["image_id"].duplicated().any():
        duplicates = frame.loc[frame["image_id"].duplicated(), "image_id"].head(5).tolist()
        raise MetadataError(f"duplicate image_id values, e.g. {duplicates}")
    for column in ("anime", "character", "split"):
        if (frame[column].astype(str).str.strip() == "").any():
            raise MetadataError(f"metadata.csv has empty {column!r} values")
    invalid_splits = sorted(set(frame["split"]) - {"train", "val", "test"})
    if invalid_splits:
        raise MetadataError(f"metadata.csv has unknown split values: {invalid_splits}")


def write_metadata_csv(frame: pd.DataFrame, path: Path | str) -> Path:
    """Project ``frame`` onto the 16 contract columns, validate, then write."""
    working = frame.copy()
    for column in METADATA_COLUMNS:
        if column not in working.columns:
            working[column] = "" if COLUMN_KINDS.get(column, "str") == "str" else 0
    projected = working.loc[:, list(METADATA_COLUMNS)]
    validate_metadata(projected)
    return write_table(projected, path, columns=METADATA_COLUMNS)


# --------------------------------------------------------------------------- #
# classes.json
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClassEntry:
    id: int
    anime: str
    character: str

    @property
    def key(self) -> str:
        return f"{self.anime}__{self.character}"


def build_classes(keys: Iterable[str]) -> List[ClassEntry]:
    """Deterministically number ``anime__character`` keys (sorted, 0-based)."""
    entries: List[ClassEntry] = []
    for index, key in enumerate(sorted({k for k in keys if k})):
        anime, _, character = key.partition("__")
        entries.append(ClassEntry(id=index, anime=anime, character=character))
    return entries


def write_classes_json(entries: Sequence[ClassEntry], path: Path | str) -> Path:
    """Write ``{"classes": [{"id": .., "anime": .., "character": ..}, ...]}``."""
    import json

    payload = {
        "classes": [
            {"id": int(entry.id), "anime": entry.anime, "character": entry.character}
            for entry in sorted(entries, key=lambda e: e.id)
        ]
    }
    return write_text_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def read_classes_json(path: Path | str) -> List[ClassEntry]:
    import json

    file_path = Path(path)
    if not file_path.is_file():
        raise MetadataError(f"classes.json not found: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise MetadataError(f"invalid classes.json {file_path}: {exc}") from exc
    classes = payload.get("classes") if isinstance(payload, Mapping) else None
    if not isinstance(classes, list):
        raise MetadataError(f"{file_path}: expected an object with a 'classes' list")
    entries = [
        ClassEntry(id=int(item["id"]), anime=str(item["anime"]), character=str(item["character"]))
        for item in classes
    ]
    return sorted(entries, key=lambda e: e.id)
