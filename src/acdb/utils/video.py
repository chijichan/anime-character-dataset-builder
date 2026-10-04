"""Video discovery, probing and frame access.

The frame-selection itself lives in the ``extract_frames`` stage; this module
only exposes the primitives:

* :func:`find_videos` - deterministic, nested-aware input discovery;
* :func:`probe_video` - fps / size / duration;
* :func:`iter_selected_frames` - one sequential decode pass that returns only
  the requested frame indices (``grab`` skips are cheap, ``retrieve`` is the
  expensive part, so we decode exactly the frames we keep);
* :func:`detect_scene_cuts` and :func:`get_keyframe_indices` - the two
  content-aware strategies, each with a dependency-free fallback so the
  pipeline still works without PySceneDetect / PyAV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

from acdb.errors import StageError
from acdb.utils.logging import get_logger
from acdb.utils.naming import video_id_for

__all__ = [
    "VideoMeta",
    "find_videos",
    "probe_video",
    "iter_selected_frames",
    "detect_scene_cuts",
    "get_keyframe_indices",
]

DEFAULT_FPS_FALLBACK = 25.0


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    video_id: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float

    def timestamp_ms(self, frame_idx: int) -> int:
        return int(round(frame_idx * 1000.0 / self.fps)) if self.fps > 0 else 0


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def find_videos(
    root: Union[str, Path],
    extensions: Sequence[str],
    *,
    recursive: bool = True,
    skip_hidden: bool = True,
    limit: Optional[int] = None,
) -> List[Path]:
    """List candidate videos under ``root`` (sorted, hidden files skipped)."""
    base = Path(root)
    if not base.exists():
        raise StageError(f"input directory does not exist: {base}")
    wanted = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    iterator: Iterable[Path] = base.rglob("*") if recursive else base.glob("*")
    found: List[Path] = []
    for candidate in iterator:
        if not candidate.is_file():
            continue
        if skip_hidden and any(part.startswith(".") for part in candidate.relative_to(base).parts):
            continue
        if candidate.suffix.lower() in wanted and not candidate.name.startswith("."):
            found.append(candidate)
    found.sort(key=lambda p: p.relative_to(base).as_posix().lower())
    if limit is not None and limit >= 0:
        found = found[:limit]
    return found


# --------------------------------------------------------------------------- #
# Probing / opening
# --------------------------------------------------------------------------- #


def _open_capture(path: Union[str, Path]) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise StageError(
            f"cannot open video {path} (unsupported codec, or a non-ASCII path on Windows)"
        )
    return capture


def probe_video(
    path: Union[str, Path],
    *,
    input_root: Optional[Union[str, Path]] = None,
    slug_maxlen: int = 40,
    hash_len: int = 8,
) -> VideoMeta:
    """Read basic metadata; ``video_id`` matches :func:`acdb.utils.naming.video_id_for`."""
    video_path = Path(path)
    capture = _open_capture(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    if fps <= 0 or not np.isfinite(fps):
        get_logger("video").warning(
            f"{video_path.name}: reported fps={fps!r}; falling back to {DEFAULT_FPS_FALLBACK}"
        )
        fps = DEFAULT_FPS_FALLBACK
    if frame_count <= 0:
        # Streams without a container index (some .ts/.flv): count by decoding.
        frame_count = _count_frames(video_path)

    return VideoMeta(
        path=video_path,
        video_id=video_id_for(video_path, input_root, slug_maxlen=slug_maxlen, hash_len=hash_len),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=frame_count / fps if fps > 0 else 0.0,
    )


def _count_frames(path: Path) -> int:
    capture = _open_capture(path)
    total = 0
    try:
        while capture.grab():
            total += 1
    finally:
        capture.release()
    return total


def iter_selected_frames(
    path: Union[str, Path],
    wanted: Set[int] | Sequence[int],
    *,
    max_index: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Decode ``path`` once, yielding ``(frame_idx, bgr)`` for wanted indices only."""
    targets = sorted({int(i) for i in wanted if i >= 0})
    if not targets:
        return
    capture = _open_capture(path)
    index = 0
    cursor = 0
    try:
        while cursor < len(targets):
            target = targets[cursor]
            if max_index is not None and target > max_index:
                break
            if not capture.grab():
                break
            if index == target:
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    yield target, frame
                cursor += 1
            index += 1
    finally:
        capture.release()


# --------------------------------------------------------------------------- #
# Scene cuts
# --------------------------------------------------------------------------- #


def detect_scene_cuts(
    path: Union[str, Path],
    *,
    threshold: float = 27.0,
    min_gap_seconds: float = 0.4,
    fps: Optional[float] = None,
) -> List[int]:
    """Frame indices that start a new shot.

    Uses PySceneDetect's ``ContentDetector`` when installed; otherwise falls
    back to a HSV-histogram comparison implemented here (documented as
    approximate in the README).
    """
    try:  # pragma: no cover - optional dependency
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(str(path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=float(threshold)))
        manager.detect_scenes(video, show_progress=False)
        cuts: List[int] = []
        for start, _end in manager.get_scene_list():
            cuts.append(int(start.get_frames()))
        return _enforce_min_gap(cuts, min_gap_seconds, fps or _safe_fps(path))
    except ImportError:
        get_logger("video").debug("PySceneDetect not installed; using the built-in scene detector")
        return _detect_scene_cuts_fallback(path, threshold=threshold,
                                           min_gap_seconds=min_gap_seconds, fps=fps)


def _enforce_min_gap(cuts: Sequence[int], min_gap_seconds: float, fps: float) -> List[int]:
    if fps <= 0:
        return sorted(set(cuts))
    min_gap = max(1, int(round(min_gap_seconds * fps)))
    kept: List[int] = []
    for cut in sorted(set(int(c) for c in cuts)):
        if not kept or cut - kept[-1] >= min_gap:
            kept.append(cut)
    return kept


def _safe_fps(path: Union[str, Path]) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    return fps if fps > 0 and np.isfinite(fps) else DEFAULT_FPS_FALLBACK


def _detect_scene_cuts_fallback(
    path: Union[str, Path],
    *,
    threshold: float,
    min_gap_seconds: float,
    fps: Optional[float] = None,
    downscale: int = 4,
) -> List[int]:
    """Histogram-difference shot boundary detector (PySceneDetect-free)."""
    capture = _open_capture(path)
    cuts: List[int] = []
    previous: Optional[np.ndarray] = None
    index = 0
    # ContentDetector thresholds live around 27 on a 0-255 scale; the histogram
    # distance below is on the same scale for typical footage.
    gate = max(1.0, float(threshold))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            small = cv2.resize(frame, (frame.shape[1] // downscale or 1,
                                      frame.shape[0] // downscale or 1),
                               interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if previous is not None:
                distance = float(cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA)) * 255.0
                if distance >= gate:
                    cuts.append(index)
            previous = hist
            index += 1
    finally:
        capture.release()
    return _enforce_min_gap(cuts, min_gap_seconds, fps or _safe_fps(path))


# --------------------------------------------------------------------------- #
# True I-frames
# --------------------------------------------------------------------------- #


def get_keyframe_indices(path: Union[str, Path]) -> Optional[List[int]]:
    """Real container keyframes (I-frames) via PyAV, or ``None`` when unreadable.

    The caller falls back to scene cuts when this returns ``None``.
    """
    try:  # pragma: no cover - optional dependency
        import av
    except ImportError:
        get_logger("video").debug("PyAV not installed; keyframes fall back to scene cuts")
        return None

    try:  # pragma: no cover - optional dependency
        import av

        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            time_base = stream.time_base
            frames: List[int] = []
            for packet in container.demux(stream):
                if packet.pts is None:
                    continue
                if packet.is_keyframe:
                    seconds = float(packet.pts * time_base)
                    frames.append(int(round(seconds * float(stream.average_rate or 0))))
            return sorted(set(f for f in frames if f >= 0))
        finally:
            container.close()
    except Exception as exc:  # pragma: no cover - codec dependent
        get_logger("video").warning(f"keyframe scan failed for {path}: {exc}")
        return None
