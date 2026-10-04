"""Crash-safe file primitives.

Rule for the whole pipeline: **never write to the final path directly**.
Encode/build the payload in memory, write it to a sibling ``*.tmp`` file, then
:func:`os.replace` it into place. On both NTFS and POSIX, ``os.replace`` is
atomic for same-volume replacements, so a killed process leaves either the old
file or the new one - never a truncated half.

``metadata.csv`` is streamed through a *temporary* path as well, because the
file can be large.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Union

__all__ = [
    "temp_sibling",
    "write_bytes_atomic",
    "write_text_atomic",
    "write_json_atomic",
    "read_json",
    "write_csv_atomic",
    "cleanup_temp_files",
]

PathLike = Union[str, Path]


def temp_sibling(path: PathLike) -> Path:
    """Return the ``*.tmp`` sibling used as the staging path for ``path``."""
    target = Path(path)
    return target.with_name(f".{target.name}.tmp")


@contextmanager
def _staged(path: PathLike) -> Iterator[Path]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = temp_sibling(target)
    try:
        yield staging
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    os.replace(staging, target)


def write_bytes_atomic(path: PathLike, data: bytes) -> Path:
    target = Path(path)
    with _staged(target) as staging:
        with open(staging, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    return target


def write_text_atomic(path: PathLike, text: str, encoding: str = "utf-8") -> Path:
    target = Path(path)
    with _staged(target) as staging:
        with open(staging, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    return target


def write_json_atomic(path: PathLike, payload: Any, *, indent: int = 2) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=False, default=str)
    return write_text_atomic(path, text + "\n")


def read_json(path: PathLike, default: Optional[Any] = None) -> Any:
    target = Path(path)
    if not target.is_file():
        return default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_csv_atomic(
    path: PathLike,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]] | Iterator[Mapping[str, Any]],
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write a CSV (no BOM, ``\\n`` line endings, LF-terminated) atomically."""
    import csv

    target = Path(path)
    with _staged(target) as staging:
        with open(staging, "w", encoding=encoding, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
    return target


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        # Keep the CSV compact but stable: 6 decimals is well below any
        # threshold we ever compare against, and avoids 17-digit float noise.
        return f"{value:.6f}".rstrip("0").rstrip(".") if value == value else ""
    return value


def cleanup_temp_files(directory: PathLike) -> List[Path]:
    """Delete leftover ``.something.tmp`` staging files (crash clean-up)."""
    removed: List[Path] = []
    for candidate in Path(directory).rglob(".*.tmp"):
        try:
            candidate.unlink()
            removed.append(candidate)
        except OSError:
            continue
    return removed
