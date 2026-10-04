"""ONNX Runtime feature extractor.

Runs any image-embedding model exported to ONNX, on CPU or CUDA, without torch::

    features:
      type: onnx
      params:
        onnx_path: models/anime_embedding.onnx
        input_size: 224
        providers: [CUDAExecutionProvider, CPUExecutionProvider]

The graph is expected to take a single ``float32`` NCHW input and to return a
feature map or a vector; ``input_name``/``output_name`` override the defaults
(the first input / the first output) when a graph has several.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from acdb.errors import ModelUnavailableError, StageError
from acdb.extractors.base import BaseFeatureExtractor
from acdb.extractors.timm_ext import prepare_batch

__all__ = ["OnnxFeatureExtractor"]


class OnnxFeatureExtractor(BaseFeatureExtractor):
    """Embeddings from an ONNX graph via onnxruntime."""

    name = "onnx"

    def __init__(self, params: Mapping[str, Any], device: str = "cpu", *, normalize: bool = True) -> None:
        super().__init__(params, device, normalize=normalize)
        try:  # pragma: no cover - optional dependency
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelUnavailableError(
                "onnxruntime is not installed. Install with\n"
                "    pip install -e \".[onnx]\"\n"
                "or use features.type=color_hist for the dependency-free path."
            ) from exc

        path = self.param("onnx_path")
        if not path:
            raise ModelUnavailableError(
                "features.params.onnx_path is required when features.type=onnx. "
                "See README -> Model weights."
            )
        self.onnx_path = str(path)

        requested = [str(p) for p in self.param("providers", ["CPUExecutionProvider"])]
        available = set(ort.get_available_providers())
        providers = [p for p in requested if p in available] or ["CPUExecutionProvider"]
        try:  # pragma: no cover - depends on the model file
            self._session = ort.InferenceSession(self.onnx_path, providers=providers)
        except Exception as exc:  # pragma: no cover - depends on the model file
            raise ModelUnavailableError(f"cannot load ONNX model {self.onnx_path}: {exc}") from exc

        self.providers = providers
        self.input_name = self.param("input_name") or self._session.get_inputs()[0].name
        outputs = self._session.get_outputs()
        self.output_name = self.param("output_name") or outputs[0].name
        self.input_size = int(self.param("input_size", 224))
        self.mean = tuple(self.param("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(self.param("std", (0.229, 0.224, 0.225)))

    # -- API --------------------------------------------------------------- #
    def describe(self) -> str:
        return f"onnx({self.onnx_path}, providers={self.providers}, input={self.input_size})"

    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        batch = prepare_batch(images, self.input_size, self.mean, self.std)
        try:  # pragma: no cover - depends on the model file
            outputs = self._session.run([self.output_name], {self.input_name: batch})
        except Exception as exc:  # pragma: no cover - depends on the model file
            raise StageError(f"ONNX inference failed: {exc}") from exc
        vectors = np.asarray(outputs[0], dtype=np.float32)
        if vectors.ndim > 2:
            vectors = vectors.reshape(vectors.shape[0], -1)
        return self._finalize(vectors)

    def close(self) -> None:
        self._session = None
