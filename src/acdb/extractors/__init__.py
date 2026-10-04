"""Feature extractor registry - ``features.type`` selects the implementation."""

from __future__ import annotations

from typing import Dict, List, Mapping, Type

from acdb.config import FeaturesConfig
from acdb.errors import ConfigError
from acdb.extractors.base import BaseFeatureExtractor, l2_normalize
from acdb.extractors.cache import (
    FeatureSet,
    compute_features,
    load_feature_cache,
    save_feature_cache,
)
from acdb.extractors.handcrafted import ColorHistogramExtractor

__all__ = [
    "EXTRACTOR_TYPES",
    "BaseFeatureExtractor",
    "FeatureSet",
    "compute_features",
    "load_feature_cache",
    "save_feature_cache",
    "l2_normalize",
    "build_extractor",
    "available_extractors",
]

#: Lazily imported backends keep heavy optional dependencies (torch, onnxruntime,
#: transformers) out of the import path unless the config actually asks for them.
_LAZY: Mapping[str, str] = {
    "timm": "acdb.extractors.timm_ext:TimmFeatureExtractor",
    "onnx": "acdb.extractors.onnx_ext:OnnxFeatureExtractor",
    "clip": "acdb.extractors.clip_ext:ClipFeatureExtractor",
    "color_hist": "acdb.extractors.handcrafted:ColorHistogramExtractor",
    "classical": "acdb.extractors.handcrafted:ColorHistogramExtractor",
}

_EAGER: Dict[str, Type[BaseFeatureExtractor]] = {
    "color_hist": ColorHistogramExtractor,
    "classical": ColorHistogramExtractor,
}


def available_extractors() -> List[str]:
    return sorted(_LAZY)


def _load(factory_path: str) -> Type[BaseFeatureExtractor]:
    module_name, _, attribute = factory_path.partition(":")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def build_extractor(config: FeaturesConfig, device: str = "cpu") -> BaseFeatureExtractor:
    """Instantiate the feature extractor named by ``config.type``."""
    name = (config.type or "").strip().lower()
    if name not in _LAZY:
        raise ConfigError(
            f"unknown features.type {config.type!r}. Available: {available_extractors()}"
        )
    factory = _EAGER.get(name) or _load(_LAZY[name])
    return factory(dict(config.params or {}), device, normalize=bool(config.normalize))
