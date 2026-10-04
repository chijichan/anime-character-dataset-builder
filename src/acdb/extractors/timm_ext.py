"""timm feature extractor (ResNet, ViT, ConvNeXt, CLIP vision towers, ...).

Any model in the timm registry works::

    features:
      type: timm
      params: {model_name: resnet18, pretrained: true, input_size: 224}

For anime-specific embeddings, point ``model_name`` at a timm-compatible
checkpoint (``hf-hub:<org>/<model>`` also works) or at a local ``.safetensors``
file. See the README for suggested backbones.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from acdb.errors import ModelUnavailableError
from acdb.extractors.base import BaseFeatureExtractor

__all__ = ["TimmFeatureExtractor", "prepare_batch", "DOWNLOAD_HINT"]

#: Shown whenever hub weights cannot be fetched. huggingface.co is unreachable
#: from mainland China (and from some corporate networks); the mirror below
#: serves the same files and is what most users there need.
DOWNLOAD_HINT = (
    "If this is a download failure, either\n"
    "  * point model_name at weights you already have (local file or directory), or\n"
    "  * use a HuggingFace mirror -- huggingface.co is not reachable from mainland\n"
    "    China, hf-mirror.com is the usual stand-in:\n"
    "        PowerShell:  $env:HF_ENDPOINT = 'https://hf-mirror.com'\n"
    "        bash:        export HF_ENDPOINT=https://hf-mirror.com\n"
    "    See README -> Model weights."
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def prepare_batch(
    images: Sequence[np.ndarray],
    input_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Resize/centre-crop/normalise BGR frames into an ``(N, 3, S, S)`` array."""
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    batch = np.empty((len(images), 3, input_size, input_size), dtype=np.float32)
    for index, image in enumerate(images):
        resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch[index] = rgb.transpose(2, 0, 1)
    return (batch - mean_array) / std_array


class TimmFeatureExtractor(BaseFeatureExtractor):
    """Penultimate-layer embeddings from any ``timm`` model."""

    name = "timm"

    def __init__(self, params: Mapping[str, Any], device: str = "cpu", *, normalize: bool = True) -> None:
        super().__init__(params, device, normalize=normalize)
        try:  # pragma: no cover - optional dependency
            import timm
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelUnavailableError(
                "timm/torch are not installed. Install with\n"
                "    pip install -e \".[torch]\"\n"
                "or use features.type=color_hist for the dependency-free path."
            ) from exc

        self._torch = torch
        self.model_name = str(self.param("model_name", "resnet18"))
        self.pretrained = bool(self.param("pretrained", True))
        self.input_size = int(self.param("input_size", 224))
        self.pool = str(self.param("pool", "avg")).lower()
        self.mean = tuple(self.param("mean", _IMAGENET_MEAN))
        self.std = tuple(self.param("std", _IMAGENET_STD))

        try:  # pragma: no cover - needs network or a warm cache
            self._model = timm.create_model(
                self.model_name, pretrained=self.pretrained, num_classes=0
            )
            self._model.eval()
            self._model.to(device)
        except Exception as exc:  # pragma: no cover - download/format errors
            raise ModelUnavailableError(
                f"cannot create timm model {self.model_name!r}: {exc}\n"
                "If this is a download failure, fetch the weights manually "
                "(see README -> Model weights)."
            ) from exc
        self.dim = int(getattr(self._model, "num_features", 0) or 0) or None

    # -- API --------------------------------------------------------------- #
    def describe(self) -> str:
        return (
            f"timm({self.model_name}, pretrained={self.pretrained}, "
            f"input={self.input_size}, device={self.device})"
        )

    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        torch = self._torch
        batch = prepare_batch(images, self.input_size, self.mean, self.std)
        with torch.no_grad():
            tensor = torch.from_numpy(batch).to(self.device)
            output = self._model(tensor)
            # Some timm models return (features, intermediates) in train mode;
            # eval mode returns a single tensor, but be defensive anyway.
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim > 2:
                output = output.mean(dim=tuple(range(2, output.ndim)))
            vectors = output.detach().cpu().numpy().astype(np.float32)
        return self._finalize(vectors)

    def close(self) -> None:
        self._model = None
