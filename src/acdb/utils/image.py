"""Image IO and the pure image-metric functions used by crop / dedup / filter.

Everything here is deliberately side-effect free (except :func:`imwrite`), so it
can be unit-tested without touching the filesystem.

Two details are worth calling out:

* **Unicode-safe IO.** ``cv2.imread``/``cv2.imwrite`` cannot open paths that
  contain characters outside the active code page on Windows, which is exactly
  what happens with Japanese file names. We therefore go through
  ``numpy.fromfile`` + ``cv2.imdecode`` and ``cv2.imencode`` + ``tofile``.
* **Atomic writes.** :func:`imwrite` encodes first, writes to a sibling
  ``*.tmp`` file and only then :func:`os.replace`\\s it into place, so a crash
  can never leave a half-written JPEG behind.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from acdb.errors import StageError

__all__ = [
    "imread",
    "imwrite",
    "laplacian_variance",
    "colorfulness",
    "exposure_ratios",
    "quality_metrics",
    "expand_bbox",
    "pad_to_square",
    "resize_long_side",
    "crop_region",
    "perception_hash",
    "hamming_distance",
    "mean_abs_diff",
    "center_square",
    "shift_into_frame",
    "background_lab",
    "saliency_distance",
    "apply_background_mask",
]

BBox = Tuple[int, int, int, int]

_PAD_MODES = {
    "edge": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT_101,
    "constant": cv2.BORDER_CONSTANT,
}


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


def imread(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image, tolerating non-ASCII paths. Raises :class:`StageError`."""
    file_path = Path(path)
    try:
        buffer = np.fromfile(str(file_path), dtype=np.uint8)
    except OSError as exc:
        raise StageError(f"cannot read image {file_path}: {exc}") from exc
    if buffer.size == 0:
        raise StageError(f"cannot read image {file_path}: file is empty")
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise StageError(f"cannot decode image {file_path}")
    return image


def imwrite(path: Union[str, Path], image: np.ndarray, quality: int = 95) -> None:
    """Atomically write ``image`` as JPEG/PNG (chosen from the suffix)."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = file_path.suffix.lower() or ".jpg"
    params: Sequence[int] = ()
    if suffix in (".jpg", ".jpeg"):
        params = (int(cv2.IMWRITE_JPEG_QUALITY), int(quality))
    elif suffix == ".png":
        params = (int(cv2.IMWRITE_PNG_COMPRESSION), 3)

    ok, buffer = cv2.imencode(suffix, image, list(params))
    if not ok:
        raise StageError(f"cannot encode image for {file_path}")

    tmp_path = file_path.with_name(file_path.name + ".tmp")
    try:
        buffer.tofile(str(tmp_path))
        os.replace(tmp_path, file_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- #
# Metrics (pure)
# --------------------------------------------------------------------------- #


def laplacian_variance(image: np.ndarray) -> float:
    """Variance of the Laplacian - the standard blur / focus measure."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def colorfulness(image: np.ndarray) -> float:
    """Hasler & Susstrunk (2003) colorfulness metric."""
    if image.ndim == 2:
        return 0.0
    b, g, r = cv2.split(image.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mean_root = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return float(std_root + 0.3 * mean_root)


def exposure_ratios(image: np.ndarray) -> Tuple[float, float]:
    """Return ``(overexposed_ratio, underexposed_ratio)`` on the V channel."""
    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
    else:
        value = image
    total = float(value.size) or 1.0
    over = float(np.count_nonzero(value >= 250)) / total
    under = float(np.count_nonzero(value <= 5)) / total
    return over, under


def quality_metrics(image: np.ndarray) -> Dict[str, float]:
    """Raw quality measurements for one crop (no thresholds applied)."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    over, under = exposure_ratios(image)
    return {
        "laplacian_var": laplacian_variance(gray),
        "overexposed_ratio": over,
        "underexposed_ratio": under,
        "colorfulness": colorfulness(image),
        "contrast": float(gray.std()),
        "brightness": float(gray.mean()),
        "width": float(image.shape[1]),
        "height": float(image.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Geometry (pure)
# --------------------------------------------------------------------------- #


def center_square(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    """Grow a box about its centre into the smallest square that contains it."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    width, height = x2 - x1, y2 - y1
    side = max(width, height)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0
    return cx - half, cy - half, cx + half, cy + half


def expand_bbox(
    bbox: Sequence[float],
    ratio: float = 1.0,
    *,
    square: bool = False,
) -> Tuple[float, float, float, float]:
    """Scale ``bbox`` about its centre by ``ratio`` and optionally squarify it."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width = (x2 - x1) * float(ratio)
    height = (y2 - y1) * float(ratio)
    box = (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
    return center_square(box) if square else box


def pad_to_square(image: np.ndarray, fill: Tuple[int, int, int] = (0, 0, 0),
                  mode: str = "edge") -> np.ndarray:
    """Pad the shorter side so the image becomes square."""
    height, width = image.shape[:2]
    if height == width:
        return image
    border_type = _PAD_MODES.get(mode, cv2.BORDER_REPLICATE)
    if width > height:
        top = (width - height) // 2
        bottom = width - height - top
        left = right = 0
    else:
        left = (height - width) // 2
        right = height - width - left
        top = bottom = 0
    value = tuple(int(v) for v in fill) if border_type == cv2.BORDER_CONSTANT else None
    return cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=value)


def resize_long_side(image: np.ndarray, long_side: Optional[int]) -> np.ndarray:
    """Downscale so that ``max(h, w) == long_side`` (never upscales)."""
    if not long_side:
        return image
    height, width = image.shape[:2]
    current = max(height, width)
    if current <= long_side:
        return image
    scale = long_side / float(current)
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def shift_into_frame(
    rect: Sequence[float], width: int, height: int
) -> Tuple[float, float, float, float]:
    """Translate ``rect`` so it lies inside the frame, when it fits.

    Padding invents pixels: with ``pad_mode: edge`` a box hanging over the
    border produced a wide smeared band of replicated edge pixels (very visible
    on tall, squared crops near a frame edge). Sliding the crop window instead
    keeps every pixel real. A rectangle larger than the frame cannot fit and is
    returned unchanged - the caller then pads, as a last resort.
    """
    x1, y1, x2, y2 = (float(v) for v in rect)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= width:
        if x1 < 0:
            x2 -= x1
            x1 = 0.0
        elif x2 > width:
            x1 -= x2 - width
            x2 = float(width)
    if box_height <= height:
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        elif y2 > height:
            y1 -= y2 - height
            y2 = float(height)
    return x1, y1, x2, y2


def crop_region(
    image: np.ndarray,
    bbox: Sequence[float],
    *,
    expand_ratio: float = 1.0,
    square: bool = False,
    pad_mode: str = "edge",
    fill: Tuple[int, int, int] = (0, 0, 0),
    fit_inside: bool = True,
) -> Tuple[np.ndarray, BBox]:
    """Crop ``bbox`` from ``image``, expanding/squarifying as configured.

    Returns ``(crop, rect)`` where ``rect`` is the realised crop rectangle **in
    source-frame coordinates**. Recording the realised rectangle (rather than
    the raw detector box) is what makes every exported image reproducible from
    ``metadata.csv`` alone.

    With ``fit_inside`` (the default) the window is kept inside the frame in two
    ways, both meant to avoid fabricated pixels:

    * a square crop is never grown beyond ``min(frame_width, frame_height)``, so
      a close-up that fills the frame yields the largest square that still fits
      instead of a padded one;
    * the window is shifted back inside the frame rather than padded whenever it
      fits (see :func:`shift_into_frame`).

    ``rect`` still reports negative/oversized coordinates in the rare case where
    the frame is smaller than the requested window; those pixels come from
    ``pad_mode``.
    """
    height, width = image.shape[:2]
    if fit_inside:
        x1, y1, x2, y2 = expand_bbox(bbox, expand_ratio, square=False)
        if square:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            side = min(max(x2 - x1, y2 - y1), float(min(width, height)))
            x1, y1, x2, y2 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
        x1, y1, x2, y2 = shift_into_frame((x1, y1, x2, y2), width, height)
    else:
        x1, y1, x2, y2 = expand_bbox(bbox, expand_ratio, square=square)
    ix1, iy1, ix2, iy2 = (int(round(v)) for v in (x1, y1, x2, y2))
    if ix2 <= ix1 or iy2 <= iy1:
        raise StageError(f"degenerate crop rectangle {(ix1, iy1, ix2, iy2)}")

    # Intersection with the frame -> real pixels; the remainder becomes padding.
    sx1, sy1 = max(ix1, 0), max(iy1, 0)
    sx2, sy2 = min(ix2, width), min(iy2, height)
    if sx2 <= sx1 or sy2 <= sy1:
        raise StageError(f"crop rectangle {(ix1, iy1, ix2, iy2)} is outside the frame")

    crop = image[sy1:sy2, sx1:sx2]
    top, left = sy1 - iy1, sx1 - ix1
    bottom, right = iy2 - sy2, ix2 - sx2
    if any((top, bottom, left, right)):
        border_type = _PAD_MODES.get(pad_mode, cv2.BORDER_REPLICATE)
        value = tuple(int(v) for v in fill) if border_type == cv2.BORDER_CONSTANT else None
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, border_type, value=value)
    return crop, (ix1, iy1, ix2, iy2)


# --------------------------------------------------------------------------- #
# Perceptual hashing (pure)
# --------------------------------------------------------------------------- #


def _dct_phash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """Reference pHash (used when ``imagehash`` is unavailable)."""
    size = hash_size * highfreq_factor
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(gray)
    low = dct[:hash_size, :hash_size]
    return low > float(np.median(low))


def perception_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """64-bit pHash as a 16-character hex string (``imagehash`` when present)."""
    try:  # pragma: no cover - depends on the optional dependency
        import imagehash
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        digest = imagehash.phash(Image.fromarray(rgb), hash_size=hash_size)
        return str(digest)
    except ImportError:
        bits = _dct_phash(image, hash_size=hash_size).reshape(-1)
        packed = np.packbits(bits.astype(np.uint8))
        return packed.tobytes().hex()


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance between two hex-encoded perceptual hashes."""
    if len(hash_a) != len(hash_b):
        raise ValueError(f"hash length mismatch: {len(hash_a)} vs {len(hash_b)}")
    try:
        distance = bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")
    except ValueError as exc:
        raise ValueError(f"invalid hex hash: {hash_a!r} / {hash_b!r}") from exc
    return distance


def mean_abs_diff(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Mean absolute per-pixel difference (0-255) between two same-shape images."""
    if image_a.shape != image_b.shape:
        raise ValueError(f"shape mismatch: {image_a.shape} vs {image_b.shape}")
    return float(np.abs(image_a.astype(np.int16) - image_b.astype(np.int16)).mean())


# --------------------------------------------------------------------------- #
# Saliency (shared by the classical detector and by background masking)
# --------------------------------------------------------------------------- #


def _odd(value: int) -> int:
    return value if value % 2 else value + 1


def background_lab(image: np.ndarray, border_frac: float = 0.08) -> np.ndarray:
    """Median CIE-Lab colour of a border ring - the background estimate."""
    blurred = image
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    band_y = max(1, int(round(height * border_frac)))
    band_x = max(1, int(round(width * border_frac)))
    ring = np.concatenate(
        [
            lab[:band_y, :, :].reshape(-1, 3),
            lab[-band_y:, :, :].reshape(-1, 3),
            lab[:, :band_x, :].reshape(-1, 3),
            lab[:, -band_x:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(ring, axis=0)


def saliency_distance(
    image: np.ndarray, *, border_frac: float = 0.08, blur_ksize: int = 5
) -> np.ndarray:
    """Per-pixel Lab distance to the estimated background colour."""
    blurred = image
    if blur_ksize and blur_ksize >= 3:
        ksize = _odd(int(blur_ksize))
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.linalg.norm(lab - background_lab(image, border_frac=border_frac), axis=2)


def apply_background_mask(
    image: np.ndarray,
    mask: np.ndarray,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Replace pixels where ``mask`` is False with a flat ``fill`` colour."""
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape[:2]} does not match image {image.shape[:2]}")
    output = image.copy()
    background = ~mask.astype(bool)
    output[background] = np.asarray(fill, dtype=output.dtype)
    return output
