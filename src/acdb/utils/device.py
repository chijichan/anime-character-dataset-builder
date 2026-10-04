"""Device selection and global seeding.

``project.device: auto`` means: CUDA when a working CUDA build of torch is
installed, then Apple MPS, then CPU. The pipeline never *requires* torch - the
classical detectors/extractors run anywhere - so every torch touch is guarded.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

from acdb.utils.logging import get_logger

__all__ = ["resolve_device", "set_global_seed", "torch_available", "describe_device"]


def torch_available() -> bool:
    """True when ``import torch`` works (cached after the first call)."""
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:  # pragma: no cover - environment dependent
            import torch  # noqa: F401

            _TORCH_AVAILABLE = True
        except Exception:  # pragma: no cover - environment dependent
            _TORCH_AVAILABLE = False
    return bool(_TORCH_AVAILABLE)


_TORCH_AVAILABLE: Optional[bool] = None


def resolve_device(spec: str = "auto") -> str:
    """Resolve ``auto|cpu|cuda|cuda:N|mps`` into a concrete torch device string."""
    spec = (spec or "auto").strip().lower()
    if spec != "auto":
        return spec
    if torch_available():  # pragma: no cover - environment dependent
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover - defensive
            pass
    return "cpu"


def describe_device(device: str) -> str:
    """Human-readable one-liner for the startup log."""
    if device.startswith("cuda") and torch_available():  # pragma: no cover
        try:
            import torch

            if torch.cuda.is_available():
                index = int(device.split(":")[1]) if ":" in device else 0
                return f"{device} ({torch.cuda.get_device_name(index)})"
        except Exception:
            pass
    return device


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and (when present) torch for reproducible splits."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if torch_available():  # pragma: no cover - environment dependent
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            get_logger().debug("torch seeding skipped")
