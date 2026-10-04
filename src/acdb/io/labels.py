"""Human-in-the-loop cluster labelling.

Two equally supported workflows, because both are natural in a file browser:

1. **Edit the YAML.** ``data/clusters/cluster_labels.yaml`` maps a cluster id to
   ``{anime}__{character}`` (or to ``unknown``)::

       clusters:
         0: demo_show_a__hikari
         1: {anime: demo_show_a, character: kaito}
         2: unknown

2. **Rename the review folder**, keeping the ``cluster_<id>`` token so the
   mapping stays unambiguous::

       data/clusters/cluster_003/                  -> unlabelled
       data/clusters/cluster_003__demo_show_b__yuki/   -> labelled
       data/clusters/demo_show_b__yuki__cluster_003/   -> labelled

Precedence: an explicit YAML value wins over a folder name; a disagreement is
logged as a warning. A folder renamed *without* the ``cluster_<id>`` token
cannot be mapped back to a cluster and is reported instead of being guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from acdb.errors import MetadataError
from acdb.utils.atomic import write_text_atomic
from acdb.utils.logging import get_logger

__all__ = [
    "LABELS_FILENAME",
    "UNKNOWN_KEY",
    "ClusterLabel",
    "cluster_folder_name",
    "labels_path",
    "cluster_dir_name_pattern",
    "write_label_template",
    "read_labels",
    "cluster_report",
]

LABELS_FILENAME = "cluster_labels.yaml"
UNKNOWN_KEY = "unknown"
CLUSTER_KEY = "clusters"

_FOLDER_RE = re.compile(r"^(?:cluster_(?P<id1>\d+)(?:__(?P<label1>.+))?"
                        r"|(?P<label2>.+?)__cluster_(?P<id2>\d+))$")


@dataclass(frozen=True)
class ClusterLabel:
    """Final label of one cluster."""

    cluster_id: int
    anime: str
    character: str
    source: str  # yaml | folder | unknown
    size: int = 0

    @property
    def key(self) -> str:
        if self.anime == UNKNOWN_KEY or not self.character:
            return UNKNOWN_KEY
        return f"{self.anime}__{self.character}"

    @property
    def is_unknown(self) -> bool:
        return self.key == UNKNOWN_KEY


def cluster_folder_name(cluster_id: int) -> str:
    """Canonical review folder name for a cluster."""
    return f"cluster_{int(cluster_id):03d}"


def labels_path(clusters_dir: Path) -> Path:
    return Path(clusters_dir) / LABELS_FILENAME


def cluster_dir_name_pattern() -> re.Pattern[str]:
    return _FOLDER_RE


# --------------------------------------------------------------------------- #
# Writing the template
# --------------------------------------------------------------------------- #


def write_label_template(
    clusters_dir: Path,
    clusters: Mapping[int, Mapping[str, Any]],
    *,
    path: Optional[Path] = None,
    overwrite: bool = True,
) -> Path:
    """Write ``cluster_labels.yaml`` pre-filled with guesses and review hints.

    ``clusters`` maps ``cluster_id`` to a small dict with at least ``size`` and
    optionally ``suggested`` (a label inferred from the dominant source video).
    Existing files are preserved unless ``overwrite`` is set, so a human's
    annotations are never clobbered by a re-run.
    """
    target = Path(path) if path is not None else labels_path(Path(clusters_dir))
    if target.is_file() and not overwrite:
        return target

    lines = [
        "# ACDB cluster labels - the human-in-the-loop step of the pipeline.",
        "#",
        "# For every cluster below, set the value to '{anime}__{character}'.",
        "# Clusters left as `unknown` (or null) are exported to unknown/ and are",
        "# dropped from the final dataset when export.drop_unknown is true.",
        "#",
        "# You can also rename the review folders instead of editing this file:",
        "#   data/clusters/cluster_003/  ->  data/clusters/cluster_003__show__name/",
        "# Keep the 'cluster_<id>' token so the folder can be mapped back.",
        "#",
        "# After editing, run:  acdb apply-labels --config <your config>",
        "",
        f"{CLUSTER_KEY}:",
    ]
    if not clusters:
        lines.append("  {}  # no cluster was found - see the log for hints")
    for cluster_id in sorted(clusters):
        info = clusters[cluster_id]
        size = int(info.get("size", 0))
        suggestion = info.get("suggested")
        hint = info.get("suggested_hint")
        preview = info.get("preview", "")
        comment = f"  # cluster {cluster_id}: {size} image(s)"
        if hint:
            comment += f", likely anime '{hint}'"
        if preview:
            comment += f", e.g. {Path(str(preview)).name}"
        lines.append(comment)
        lines.append(f"  {cluster_id}: {suggestion or UNKNOWN_KEY}")
    payload = "\n".join(lines) + "\n"
    return write_text_atomic(target, payload)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _parse_value(value: Any) -> Optional[Tuple[str, str]]:
    """Turn a YAML value into ``(anime, character)``; ``None`` means unknown."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        anime = str(value.get("anime", "") or "").strip()
        character = str(value.get("character", "") or "").strip()
        if not anime or not character:
            return None
        return anime, character
    text = str(value).strip()
    if not text or text.lower() in {UNKNOWN_KEY, "none", "null", "-1"}:
        return None
    if "__" not in text:
        raise MetadataError(
            f"cluster label {text!r} must be '{{anime}}__{{character}}' or 'unknown'"
        )
    anime, _, character = text.partition("__")
    anime, character = anime.strip(), character.strip()
    if not anime or not character:
        raise MetadataError(f"cluster label {text!r} has an empty anime or character part")
    return anime, character


def _folder_labels(clusters_dir: Path) -> Dict[int, Tuple[Optional[Tuple[str, str]], str]]:
    """Infer labels from renamed review folders."""
    found: Dict[int, Tuple[Optional[Tuple[str, str]], str]] = {}
    if not clusters_dir.is_dir():
        return found
    for entry in sorted(clusters_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = _FOLDER_RE.match(entry.name)
        if not match:
            continue
        raw_id = match.group("id1") or match.group("id2")
        raw_label = match.group("label1") or match.group("label2")
        cluster_id = int(raw_id)
        if raw_label is None:
            continue
        found[cluster_id] = (_parse_value(raw_label), entry.name)
    return found


def _yaml_labels(path: Path) -> Dict[int, Tuple[Optional[Tuple[str, str]], str]]:
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError) as exc:
        raise MetadataError(f"invalid {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise MetadataError(f"{path}: top level must be a mapping with a '{CLUSTER_KEY}' key")
    block = payload.get(CLUSTER_KEY, payload)
    if block is None:
        # An empty `clusters:` key (or a file with everything commented out) is
        # a legitimate "nothing labelled yet" state, not an error.
        return {}
    if not isinstance(block, Mapping):
        raise MetadataError(f"{path}: '{CLUSTER_KEY}' must be a mapping of id -> label")
    result: Dict[int, Tuple[Optional[Tuple[str, str]], str]] = {}
    for key, value in block.items():
        try:
            cluster_id = int(key)
        except (TypeError, ValueError) as exc:
            raise MetadataError(f"{path}: cluster id {key!r} is not an integer") from exc
        result[cluster_id] = (_parse_value(value), LABELS_FILENAME)
    return result


def read_labels(clusters_dir: Path, *, log: bool = True) -> Dict[int, ClusterLabel]:
    """Merge YAML and folder labels into ``{cluster_id: ClusterLabel}``."""
    directory = Path(clusters_dir)
    logger = get_logger("apply_labels")
    yaml_map = _yaml_labels(labels_path(directory))
    folder_map = _folder_labels(directory)

    labels: Dict[int, ClusterLabel] = {}
    for cluster_id in sorted(set(yaml_map) | set(folder_map)):
        yaml_entry = yaml_map.get(cluster_id)
        folder_entry = folder_map.get(cluster_id)
        # An explicit YAML value (including `unknown`) beats a folder rename.
        if yaml_entry is not None and yaml_entry[0] is not None:
            parsed, source = yaml_entry
        elif folder_entry is not None:
            parsed, source = folder_entry
        elif yaml_entry is not None:
            parsed, source = yaml_entry
        else:  # pragma: no cover - unreachable
            parsed, source = None, UNKNOWN_KEY

        if (
            yaml_entry is not None
            and folder_entry is not None
            and yaml_entry[0] is not None
            and folder_entry[0] is not None
            and yaml_entry[0] != folder_entry[0]
        ):
            if log:
                logger.warning(
                    f"cluster {cluster_id}: {LABELS_FILENAME} says "
                    f"{yaml_entry[0][0]}__{yaml_entry[0][1]} but folder "
                    f"{folder_entry[1]} says {folder_entry[0][0]}__{folder_entry[0][1]}; "
                    f"keeping the YAML value"
                )
        anime, character = parsed if parsed else (UNKNOWN_KEY, UNKNOWN_KEY)
        labels[cluster_id] = ClusterLabel(
            cluster_id=cluster_id, anime=anime, character=character, source=source
        )
    return labels


def cluster_report(clusters_dir: Path, counts: Mapping[int, int]) -> Dict[str, Any]:
    """Summary dict used by the CLI's ``status``/``cluster`` output."""
    labels = read_labels(clusters_dir, log=False)
    labelled = {cid: label for cid, label in labels.items() if not label.is_unknown}
    return {
        "clusters": len(counts),
        "labelled": len(labelled),
        "unknown": len(counts) - len(labelled),
        "images": int(sum(counts.values())),
    }
