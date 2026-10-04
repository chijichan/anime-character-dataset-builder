"""Deterministic, filesystem-safe artefact naming.

This module owns the *hard output contract* for file names:

======================  ==========================================================
frames                  ``{video_id}_f{frame_idx:06d}.jpg``
crops                   ``{video_id}_f{frame_idx:06d}_b{bbox_idx:02d}.jpg``
final dataset images    ``{anime}__{character}__{video_id}_f{frame_idx:06d}_b{bbox_idx:02d}.jpg``
=========================  ========================================================

``video_id`` derivation
-----------------------
``"{slug}_{hash}"`` where

* ``slug`` is the ASCII-folded, lowercased, underscore-collapsed stem of the
  video file, truncated to ``video_id_slug_maxlen`` characters. Runs of
  underscores are collapsed to a single one so that a slug can never contain
  the ``__`` separator used between ``anime`` and ``character``; non-ASCII
  characters are dropped (a Japanese file name therefore contributes an empty
  slug), which keeps the identifier portable across Windows/macOS/Linux and
  avoids Unicode-normalisation mismatches.
* ``hash`` is the first ``video_id_hash_len`` hex digits of the BLAKE2b digest
  of the video's *path relative to the input directory* (POSIX separators,
  lowercased). Two files with the same stem in different folders therefore get
  different ids, while moving the whole input tree keeps every id stable.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Union

__all__ = [
    "slugify",
    "video_id_for",
    "frame_filename",
    "crop_filename",
    "final_filename",
    "class_dirname",
    "parse_final_filename",
    "frame_filename_from_crop",
]

_NON_SAFE = re.compile(r"[^0-9a-z]+")
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def slugify(text: str, maxlen: int = 40, fallback: str = "video") -> str:
    """ASCII-fold ``text`` into ``[a-z0-9_]`` with single underscores.

    Returns ``fallback`` when nothing usable survives (e.g. a fully
    non-ASCII name). The result never contains ``__`` and never starts or
    ends with ``_``, so it is safe to embed in a ``__``-delimited file name.
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _NON_SAFE.sub("_", folded.lower()).strip("_")
    if maxlen > 0:
        slug = slug[:maxlen].strip("_")
    if not slug:
        slug = fallback
    if slug in _WINDOWS_RESERVED:
        slug = f"{slug}_v"
    return slug


def video_id_for(
    video_path: Union[str, Path],
    input_root: Optional[Union[str, Path]] = None,
    *,
    slug_maxlen: int = 40,
    hash_len: int = 8,
) -> str:
    """Build the stable ``video_id`` for ``video_path`` (see module docstring)."""
    path = Path(video_path)
    if input_root is not None:
        try:
            key = path.resolve().relative_to(Path(input_root).resolve()).as_posix()
        except ValueError:
            key = path.name
    else:
        key = path.name
    digest = hashlib.blake2b(key.lower().encode("utf-8"), digest_size=16).hexdigest()[:hash_len]
    return f"{slugify(path.stem, maxlen=slug_maxlen)}_{digest}"


def class_dirname(anime: str, character: str) -> str:
    """``{anime}__{character}`` - the ImageFolder class directory name."""
    return f"{anime}__{character}"


def frame_filename(video_id: str, frame_idx: int, template: str) -> str:
    return template.format(video_id=video_id, frame_idx=int(frame_idx))


def crop_filename(video_id: str, frame_idx: int, bbox_idx: int, template: str) -> str:
    return template.format(video_id=video_id, frame_idx=int(frame_idx), bbox_idx=int(bbox_idx))


def frame_filename_from_crop(crop_name: str, crop_template: str, frame_template: str) -> str:
    """Map a crop file name back to its source frame file name."""
    pattern = re.escape(crop_template).replace(
        re.escape("{video_id}"), r"(?P<video_id>.+?)"
    ).replace(
        re.escape("{frame_idx:06d}"), r"(?P<frame_idx>\d+)"
    ).replace(
        re.escape("{bbox_idx:02d}"), r"(?P<bbox_idx>\d+)"
    )
    match = re.match(pattern + "$", crop_name)
    if not match:
        raise ValueError(f"{crop_name!r} does not match crop template {crop_template!r}")
    return frame_filename(
        match.group("video_id"), int(match.group("frame_idx")), frame_template
    )


def final_filename(
    anime: str,
    character: str,
    video_id: str,
    frame_idx: int,
    bbox_idx: int,
    template: str,
) -> str:
    return template.format(
        anime=anime,
        character=character,
        video_id=video_id,
        frame_idx=int(frame_idx),
        bbox_idx=int(bbox_idx),
    )


def parse_final_filename(name: str) -> Optional[Dict[str, Any]]:
    """Parse ``{anime}__{character}__{video_id}_f{frame:06d}_b{bbox:02d}.jpg``.

    Returns ``None`` when the name does not follow the contract; otherwise a
    dict with ``anime``, ``character``, ``video_id``, ``frame_idx`` and
    ``bbox_idx``. Splitting happens from the left with ``maxsplit=2`` so that
    ``__`` inside a character name (which the contract forbids anyway) could
    not silently shift the fields. Only the JPEG extensions the pipeline writes
    are accepted, so this doubles as a contract validator.
    """
    path = Path(name)
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return None
    stem = path.stem
    parts = stem.split("__", 2)
    if len(parts) != 3:
        return None
    anime, character, rest = parts
    match = re.match(r"^(?P<video_id>.+)_f(?P<frame_idx>\d+)_b(?P<bbox_idx>\d+)$", rest)
    if not match:
        return None
    return {
        "anime": anime,
        "character": character,
        "video_id": match.group("video_id"),
        "frame_idx": int(match.group("frame_idx")),
        "bbox_idx": int(match.group("bbox_idx")),
    }
