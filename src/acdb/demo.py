"""Demo-only helpers: synthesise footage and simulate the human labelling step.

These helpers exist so that ``docs/demo_run.md`` is reproducible on any machine
without downloading a single model weight or a single frame of copyrighted
footage:

* :func:`make_demo_videos` renders three short synthetic "episodes", each with
  two flat-shaded characters whose hair/uniform palettes are the only identity
  signal - exactly what a colour-histogram extractor can separate and what the
  classical saliency detector can localise.
* :func:`simulate_cluster_labels` plays the role of the human annotator by
  matching each review folder's dominant palette against the palette the
  generator recorded. It writes ``cluster_labels.yaml``; nothing else in the
  pipeline knows or cares that the labels were produced by a script.

Both are deterministic given the same seed, so the demo run is reproducible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from acdb.errors import StageError
from acdb.io import write_json_atomic, write_text_atomic
from acdb.utils.image import imread
from acdb.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from acdb.extractors.base import BaseFeatureExtractor

__all__ = [
    "DEMO_CHARACTERS",
    "make_demo_videos",
    "simulate_cluster_labels",
    "CharacterSpec",
]


@dataclass(frozen=True)
class CharacterSpec:
    """Palette and motion of one synthetic character."""

    anime: str
    character: str
    hair: Tuple[int, int, int]     # BGR
    outfit: Tuple[int, int, int]   # BGR
    accent: Tuple[int, int, int]   # BGR
    lane: float                    # vertical position (0-1)
    speed: float                   # cycles across the frame
    phase: float                   # start offset (0-1)
    scale: float = 1.0


#: Six characters over three synthetic shows; palettes are deliberately far
#: apart in hue so the demo's failure modes are about the *pipeline*, not about
#: an ambiguous ground truth.
DEMO_CHARACTERS: Sequence[CharacterSpec] = (
    CharacterSpec("demo_show_a", "hikari", (182, 127, 255), (255, 255, 255), (120, 220, 255), 0.30, 1.0, 0.00),
    CharacterSpec("demo_show_a", "kaito", (255, 91, 46), (34, 34, 34), (200, 200, 200), 0.72, 1.3, 0.55),
    CharacterSpec("demo_show_b", "yuki", (245, 237, 237), (208, 192, 136), (90, 90, 120), 0.28, 0.8, 0.20),
    CharacterSpec("demo_show_b", "ren", (80, 175, 76), (51, 51, 51), (240, 240, 240), 0.70, 1.1, 0.75),
    CharacterSpec("demo_show_c", "sora", (28, 159, 255), (255, 255, 255), (60, 200, 240), 0.32, 1.2, 0.35),
    CharacterSpec("demo_show_c", "mei", (229, 93, 155), (43, 43, 43), (200, 160, 255), 0.74, 0.9, 0.85),
)

_BACKGROUND_TOP = (74, 46, 42)      # BGR, dark indigo
_BACKGROUND_BOTTOM = (102, 64, 58)  # BGR
_SKIN = (189, 224, 255)             # BGR


def _palette_for(anime: str) -> List[CharacterSpec]:
    return [spec for spec in DEMO_CHARACTERS if spec.anime == anime]


def _draw_character(frame: np.ndarray, spec: CharacterSpec, x: float, y: float, scale: float) -> None:
    """Draw a stylised flat-shaded figure (head, hair cap, torso, limbs)."""
    height = int(round(110 * scale))
    head_radius = max(6, int(height * 0.22))
    body_height = int(height * 0.52)
    body_width = int(height * 0.34)
    centre_x, centre_y = int(round(x)), int(round(y))

    # torso
    top_left = (centre_x - body_width // 2, centre_y)
    bottom_right = (centre_x + body_width // 2, centre_y + body_height)
    cv2.rectangle(frame, top_left, bottom_right, spec.outfit, thickness=cv2.FILLED)
    # arms
    cv2.line(frame, (top_left[0], centre_y + 6), (top_left[0] - body_width // 3, centre_y + body_height - 4),
             spec.outfit, thickness=max(3, body_width // 4))
    cv2.line(frame, (bottom_right[0], centre_y + 6), (bottom_right[0] + body_width // 3, centre_y + body_height - 4),
             spec.outfit, thickness=max(3, body_width // 4))
    # head + hair + eyes + accent
    head_centre = (centre_x, centre_y - head_radius)
    cv2.circle(frame, head_centre, head_radius, _SKIN, thickness=cv2.FILLED)
    cv2.ellipse(
        frame,
        head_centre,
        (head_radius, head_radius),
        0, 180, 360, spec.hair, thickness=cv2.FILLED,
    )
    cv2.ellipse(
        frame,
        (centre_x, centre_y - head_radius - 2),
        (head_radius, max(3, head_radius // 2)),
        0, 180, 360, spec.hair, thickness=cv2.FILLED,
    )
    eye_offset = max(2, head_radius // 3)
    eye_radius = max(1, head_radius // 7)
    for direction in (-1, 1):
        cv2.circle(
            frame,
            (centre_x + direction * eye_offset, centre_y - head_radius + eye_radius),
            eye_radius,
            (40, 40, 40),
            thickness=cv2.FILLED,
        )
    cv2.rectangle(
        frame,
        (centre_x - body_width // 2, centre_y + 2),
        (centre_x + body_width // 2, centre_y + max(3, body_height // 5)),
        spec.accent,
        thickness=cv2.FILLED,
    )


def _render_frame(width: int, height: int, t: float, specs: Sequence[CharacterSpec]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        ratio = row / max(1, height - 1)
        frame[row, :] = [
            int(_BACKGROUND_TOP[channel] * (1 - ratio) + _BACKGROUND_BOTTOM[channel] * ratio)
            for channel in range(3)
        ]
    # static scenery so the border ring is a stable background estimate
    cv2.circle(frame, (int(width * 0.12), int(height * 0.18)), max(6, height // 16),
               (96, 78, 60), thickness=cv2.FILLED)
    cv2.rectangle(frame, (int(width * 0.80), int(height * 0.08)),
                  (int(width * 0.92), int(height * 0.24)), (70, 58, 48), thickness=cv2.FILLED)
    cv2.line(frame, (0, int(height * 0.88)), (width, int(height * 0.90)), (120, 84, 70),
             thickness=max(2, height // 60))

    for spec in specs:
        progress = (t * spec.speed + spec.phase) % 1.0
        x = width * (0.16 + 0.68 * progress)
        bob = math.sin((t * 2.4 + spec.phase) * math.tau) * height * 0.02
        y = height * spec.lane + bob
        _draw_character(frame, spec, x, y, scale=spec.scale * (height / 360.0))
    return frame


def _render_reference(
    spec: CharacterSpec,
    size: int,
    background: Tuple[np.ndarray, np.ndarray],
    view: str = "full",
) -> np.ndarray:
    """Render one character on a square canvas shaped like a real crop.

    Two views are produced per character, mirroring the multi-view character
    sheets a human annotator consults: ``full`` (whole figure - the common
    case) and ``head`` (face close-up, which is what a crop looks like when the
    detector only locks onto the head).
    """
    top, bottom = background
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    for row in range(size):
        ratio = row / max(1, size - 1)
        canvas[row, :] = [
            int(top[channel] * (1 - ratio) + bottom[channel] * ratio) for channel in range(3)
        ]
    if view == "head":
        scale = (0.34 * size) / (110.0 * 0.22)  # head radius == 34% of the canvas
        y = size * 0.45 + 110.0 * scale * 0.22  # `y` is where the torso starts
    else:
        scale = (0.62 * size) / (110.0 * 0.74)
        y = size * 0.86 - 110.0 * scale * 0.52
    _draw_character(canvas, spec, size / 2.0, y, scale=scale)
    return canvas


def make_demo_videos(
    output_dir: Path,
    *,
    width: int = 640,
    height: int = 360,
    fps: int = 12,
    seconds: float = 12.0,
    episodes: int = 3,
    shows: Optional[Sequence[str]] = None,
    fourcc: str = "mp4v",
    seed: int = 20240517,
) -> Dict[str, object]:
    """Render the synthetic demo episodes, reference sheets and ground truth.

    ``episodes`` files are rendered per show, each with the characters at a
    different point in their paths. Three of them is deliberate: it gives every
    class more than one source video, so the *strict* leak-free split
    (``split.prevent_leakage: true``, one ``(class, video)`` unit per split) can
    fill train/val/test without falling back to per-image assignment.

    Returns a dict with ``videos`` (played-back path -> characters) and
    ``palette`` (character key -> hair/outfit BGR). The per-character
    ``reference/<anime>__<character>__<view>.jpg`` sheets are what the simulated
    annotator matches against - the stand-in for a human who knows what each
    character looks like.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reference_dir = root / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shows = list(shows) if shows else sorted({spec.anime for spec in DEMO_CHARACTERS})
    episodes = max(1, int(episodes))
    total_frames = int(round(fps * seconds))
    manifest: Dict[str, object] = {"videos": {}, "palette": {}, "reference": {}}
    background = (np.asarray(_BACKGROUND_TOP), np.asarray(_BACKGROUND_BOTTOM))

    for anime in shows:
        specs = _palette_for(anime)
        if not specs:
            raise StageError(f"no demo character defined for show {anime!r}")
        show_dir = root / anime
        show_dir.mkdir(parents=True, exist_ok=True)

        for episode in range(episodes):
            target = show_dir / f"ep{episode + 1:02d}.mp4"
            writer = cv2.VideoWriter(
                str(target), cv2.VideoWriter_fourcc(*fourcc), float(fps), (width, height)
            )
            if not writer.isOpened():  # pragma: no cover - codec dependent
                raise StageError(
                    f"cannot open a VideoWriter for {target} (codec {fourcc!r} unavailable)"
                )
            # Each episode samples the characters' paths from a different point,
            # so consecutive episodes are not near-duplicates of each other.
            offset = episode * 2.7
            try:
                for index in range(total_frames):
                    frame = _render_frame(width, height, index / fps + offset, specs)
                    noise = rng.normal(0.0, 1.6, frame.shape[:2])[:, :, None]
                    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                    writer.write(frame)
            finally:
                writer.release()
            manifest["videos"][f"{anime}/{target.name}"] = [  # type: ignore[index]
                {"character": spec.character, "hair": list(spec.hair), "outfit": list(spec.outfit)}
                for spec in specs
            ]
            get_logger("demo").info(
                f"wrote {target} ({total_frames} frames, {width}x{height} @ {fps}fps)"
            )

        for spec in specs:
            key = f"{spec.anime}__{spec.character}"
            views: Dict[str, str] = {}
            for view in ("full", "head"):
                sheet = _render_reference(spec, 256, background, view=view)
                sheet_path = reference_dir / f"{key}__{view}.jpg"
                cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                views[view] = sheet_path.relative_to(root).as_posix()
            manifest["reference"][key] = views  # type: ignore[index]

        manifest["palette"][anime] = {  # type: ignore[index]
            spec.character: {"hair": list(spec.hair), "outfit": list(spec.outfit)}
            for spec in specs
        }

    manifest["meta"] = {  # type: ignore[index]
        "width": width,
        "height": height,
        "fps": fps,
        "seconds": seconds,
        "episodes": episodes,
        "seed": seed,
    }
    write_json_atomic(root / "demo_ground_truth.json", manifest)
    return manifest


# --------------------------------------------------------------------------- #
# Simulated annotator
# --------------------------------------------------------------------------- #


def _hinted_anime(folder: Path) -> str:
    """Read the dominant show hint that `cluster` wrote into ``index.csv``."""
    index = folder / "index.csv"
    if not index.is_file():
        return ""
    try:
        import csv

        with open(index, encoding="utf-8", newline="") as handle:
            counts: Dict[str, int] = {}
            for row in csv.DictReader(handle):
                anime = (row.get("suggested_anime") or "").strip()
                if anime:
                    counts[anime] = counts.get(anime, 0) + 1
    except OSError:  # pragma: no cover - defensive
        return ""
    return max(counts, key=counts.get) if counts else ""


def _folder_images(folder: Path, limit: int) -> List[Path]:
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:limit]


def simulate_cluster_labels(
    clusters_dir: Path,
    ground_truth_path: Path,
    extractor: BaseFeatureExtractor,
    *,
    overwrite: bool = True,
    min_similarity: float = 0.80,
    limit: int = 16,
) -> Dict[str, object]:
    """Write ``cluster_labels.yaml`` the way a human would, but automatically.

    A human opens a review folder, recognises the character and writes
    ``{anime}__{character}``. This function does the same thing with the same
    tooling as the pipeline: it embeds the representative images with the
    configured feature extractor, embeds the per-character *reference sheets*
    that ``acdb demo-video`` rendered, and picks the closest match - restricted
    to the show the folder's ``index.csv`` points at, exactly like a human who
    knows which episode they are labelling.

    Below ``min_similarity`` a cluster stays ``unknown`` rather than being
    guessed, which is also what the pipeline does with unlabelled clusters.
    """


    directory = Path(clusters_dir)
    if not directory.is_dir():
        raise StageError(f"no cluster review folder at {directory}; run `acdb cluster` first")
    truth_file = Path(ground_truth_path)
    if not truth_file.is_file():
        raise StageError(f"ground-truth manifest not found: {truth_file}")
    truth = json.loads(truth_file.read_text(encoding="utf-8"))
    reference = truth.get("reference", {})
    if not reference:
        raise StageError(
            f"{truth_file} has no 'reference' entries - regenerate it with "
            f"`acdb demo-video --force`"
        )
    base_dir = Path(ground_truth_path).parent
    # One embedding per (character, view); a cluster matches the best view.
    reference_keys: List[str] = []
    reference_vectors: List[np.ndarray] = []
    for key in sorted(reference):
        views = reference[key]
        paths = [views[view] for view in sorted(views)] if isinstance(views, dict) else [views]
        vectors = extractor.extract([imread(base_dir / path) for path in paths])
        reference_keys.append(key)
        reference_vectors.append(vectors)
    reference_stack = np.stack(reference_vectors)

    labels: Dict[int, str] = {}
    details: List[dict] = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("cluster_"):
            continue
        cluster_id = int(entry.name[len("cluster_"):].split("__")[0])
        images = _folder_images(entry, limit)
        if not images:
            labels[cluster_id] = "unknown"
            details.append({"cluster": cluster_id, "label": "unknown", "reason": "no images"})
            continue

        vectors = extractor.extract([imread(path) for path in images])
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid)) or 1.0
        centroid = centroid / norm
        # Max over views, then over nothing else: keep the best view per character.
        per_view = np.einsum("kvd,d->kv", reference_stack, centroid)
        per_view = per_view / np.maximum(
            np.linalg.norm(reference_stack, axis=2), 1e-12
        )
        similarities = per_view.max(axis=1)

        hint = _hinted_anime(entry)
        order = np.argsort(-similarities)
        ranked = [
            (float(similarities[index]), reference_keys[index])
            for index in order
            if not hint or reference_keys[index].split("__", 1)[0] == hint
        ]
        if not ranked:
            ranked = [(float(similarities[index]), reference_keys[index]) for index in order]
        best_similarity, best_key = ranked[0]
        if best_similarity < min_similarity:
            labels[cluster_id] = "unknown"
            details.append(
                {
                    "cluster": cluster_id,
                    "label": "unknown",
                    "reason": f"best similarity {best_similarity:.3f} < {min_similarity}",
                }
            )
            continue

        labels[cluster_id] = best_key
        details.append(
            {
                "cluster": cluster_id,
                "label": best_key,
                "similarity": round(best_similarity, 3),
                "anime_hint": hint,
                "runner_up": ranked[1][1] if len(ranked) > 1 else "",
                "runner_up_similarity": round(ranked[1][0], 3) if len(ranked) > 1 else None,
            }
        )

    target = directory / "cluster_labels.yaml"
    if target.is_file() and not overwrite:
        return {"path": str(target), "written": False, "details": details}

    lines = [
        "# Written by `acdb demo-annotate` (simulated human annotator).",
        "# Replace these values with real labels when you run on real footage.",
        "clusters:",
    ]
    for cluster_id in sorted(labels):
        lines.append(f"  {cluster_id}: {labels[cluster_id]}")
    write_text_atomic(target, "\n".join(lines) + "\n")
    get_logger("demo").info(
        f"simulated annotator labelled {sum(1 for v in labels.values() if v != 'unknown')}"
        f"/{len(labels)} cluster(s) -> {target}"
    )
    return {"path": str(target), "written": True, "labels": labels, "details": details}
