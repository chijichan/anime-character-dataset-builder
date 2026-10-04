"""CLIP image embeddings (``transformers`` backend).

Cosine similarity in CLIP space is a strong identity signal for stylised
characters, which makes it a good default for the dedup fine gate and for
clustering::

    features:
      type: clip
      params:
        model_name: openai/clip-vit-base-patch32

Only the vision tower is used, so no text encoder is loaded.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from acdb.errors import ModelUnavailableError
from acdb.extractors.base import BaseFeatureExtractor
from acdb.extractors.timm_ext import prepare_batch

__all__ = ["ClipFeatureExtractor"]

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipFeatureExtractor(BaseFeatureExtractor):
    """Image embeddings from a HuggingFace CLIP vision tower."""

    name = "clip"

    def __init__(self, params: Mapping[str, Any], device: str = "cpu", *, normalize: bool = True) -> None:
        super().__init__(params, device, normalize=normalize)
        try:  # pragma: no cover - optional dependency
            import torch
            from transformers import CLIPVisionModelWithProjection
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelUnavailableError(
                "transformers/torch are not installed. Install with\n"
                "    pip install transformers torch\n"
                "or use features.type=timm / color_hist instead."
            ) from exc

        self._torch = torch
        self.model_name = str(self.param("model_name", "openai/clip-vit-base-patch32"))
        self.input_size = int(self.param("input_size", 224))
        try:  # pragma: no cover - needs network or a warm cache
            self._model = CLIPVisionModelWithProjection.from_pretrained(self.model_name)
            self._model.eval().to(device)
        except Exception as exc:  # pragma: no cover - download errors
            raise ModelUnavailableError(
                f"cannot load CLIP model {self.model_name!r}: {exc}"
            ) from exc
        self.dim = int(getattr(self._model.config, "projection_dim", 0) or 0) or None

    # -- API --------------------------------------------------------------- #
    def describe(self) -> str:
        return f"clip({self.model_name}, device={self.device})"

    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        torch = self._torch
        batch = prepare_batch(images, self.input_size, _CLIP_MEAN, _CLIP_STD)
        with torch.no_grad():
            tensor = torch.from_numpy(batch).to(self.device)
            output = self._model(pixel_values=tensor)
            vectors = output.image_embeds.detach().cpu().numpy().astype(np.float32)
        return self._finalize(vectors)

    def close(self) -> None:
        self._model = None
