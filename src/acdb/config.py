"""Typed configuration loading, validation and hashing.

Design notes
------------
* The YAML tree is materialised into frozen dataclasses so that every stage
  gets attribute access with real type annotations instead of ``dict["..."]``.
* **Unknown keys are a hard error.** A typo such as ``min_conf`` vs
  ``min_confidence`` fails loudly at startup rather than silently disabling a
  filter for the whole run.
* One level of inheritance is supported through the top-level ``base:`` key,
  deep-merged with the current document (the current document wins). The demo
  config uses it to stay short while remaining fully explicit.
* Relative paths resolve against *the project root* (``--root``, default: the
  current working directory), never against the config file, so moving a config
  between directories cannot silently retarget the pipeline.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

from acdb.errors import ConfigError

__all__ = [
    "AppConfig",
    "load_config",
    "apply_overrides",
    "deep_merge",
    "derive_work_dir",
    "hash_mapping",
]

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "anime-character-dataset"
    seed: int = 1337
    device: str = "auto"
    max_videos: Optional[int] = None


@dataclass(frozen=True)
class PathsConfig:
    input_dir: Path = Path("data/input")
    work_dir: Path = Path("data")
    frames_dir: Path = Path("data/frames")
    crops_dir: Path = Path("data/crops")
    rejected_dir: Path = Path("data/rejected")
    dataset_dir: Path = Path("data/dataset")
    metadata_dir: Path = Path("data/metadata")
    clusters_dir: Path = Path("data/clusters")
    logs_dir: Path = Path("logs")

    def state_dir(self) -> Path:
        return self.metadata_dir / ".state"

    def output_dirs(self) -> Tuple[Path, ...]:
        return (
            self.frames_dir,
            self.crops_dir,
            self.rejected_dir,
            self.dataset_dir,
            self.metadata_dir,
            self.clusters_dir,
            self.logs_dir,
        )


@dataclass(frozen=True)
class NamingConfig:
    frame_template: str = "{video_id}_f{frame_idx:06d}.jpg"
    crop_template: str = "{video_id}_f{frame_idx:06d}_b{bbox_idx:02d}.jpg"
    final_template: str = "{anime}__{character}__{video_id}_f{frame_idx:06d}_b{bbox_idx:02d}.jpg"
    video_id_slug_maxlen: int = 40
    video_id_hash_len: int = 8


@dataclass(frozen=True)
class VideoConfig:
    extensions: Tuple[str, ...] = (
        ".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv", ".m4v", ".ts", ".wmv",
    )
    recursive: bool = True
    skip_hidden: bool = True


@dataclass(frozen=True)
class ExtractFramesConfig:
    strategies: Tuple[str, ...] = ("interval", "scene")
    interval_seconds: float = 1.0
    interval_frames: Optional[int] = None
    scene_threshold: float = 27.0
    scene_min_gap_seconds: float = 0.4
    keyframe_backend: str = "auto"
    max_frames_per_video: Optional[int] = None
    resize_long_side: Optional[int] = None
    jpeg_quality: int = 95
    skip_static: bool = True
    static_diff_threshold: float = 0.5


@dataclass(frozen=True)
class DetectFilterConfig:
    min_conf: float = 0.35
    min_side: int = 48
    max_side: Optional[int] = None
    aspect_ratio: Tuple[float, float] = (0.2, 5.0)
    min_area_ratio: float = 0.0015
    max_per_frame: int = 10
    nms_across_frames: bool = False


@dataclass(frozen=True)
class DetectConfig:
    type: str = "yolo_v8"
    params: Mapping[str, Any] = field(default_factory=dict)
    filter: DetectFilterConfig = field(default_factory=DetectFilterConfig)
    #: Frames per inference call (batched backends only; harmless otherwise).
    batch_size: int = 8


@dataclass(frozen=True)
class CropConfig:
    expand_ratio: float = 1.2
    square: bool = True
    keep_background: bool = True
    background_fill: Tuple[int, int, int] = (0, 0, 0)
    background_min_distance: float = 26.0
    pad_mode: str = "edge"
    min_output_side: int = 32
    output_size: Optional[int] = None
    jpeg_quality: int = 95


@dataclass(frozen=True)
class DedupConfig:
    phash_size: int = 8
    phash_hamming: int = 6
    cosine_threshold: float = 0.985
    keep: str = "best_quality"
    same_video_only: bool = False
    max_group_size: int = 10


@dataclass(frozen=True)
class FeaturesConfig:
    type: str = "timm"
    normalize: bool = True
    batch_size: int = 32
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityWeights:
    sharpness: float = 0.45
    exposure: float = 0.25
    contrast: float = 0.15
    colorfulness: float = 0.15


@dataclass(frozen=True)
class QualityConfig:
    enabled: bool = True
    min_laplacian_var: float = 60.0
    min_side_px: int = 64
    max_overexposed_ratio: float = 0.35
    max_underexposed_ratio: float = 0.60
    min_colorfulness: float = 4.0
    min_score: float = 0.25
    weights: QualityWeights = field(default_factory=QualityWeights)
    move_rejected: bool = True


@dataclass(frozen=True)
class ClusterConfig:
    type: str = "hdbscan"
    params: Mapping[str, Any] = field(default_factory=dict)
    reduce_dim: str = "pca"
    pca_components: int = 32
    representatives_per_cluster: int = 12
    noise_as_unknown: bool = True


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def as_dict(self) -> Dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True)
class SplitConfig:
    ratios: SplitRatios = field(default_factory=SplitRatios)
    seed: int = 1337
    stratify: bool = True
    prevent_leakage: bool = True
    min_images_per_class: int = 2
    #: What to do when a class is confined to too few videos to fill every
    #: non-empty split with whole ``(class, video)`` units:
    #: ``image`` degrades that class to per-image assignment (warning logged),
    #: ``train_only`` keeps the leak guarantee and puts the class in train.
    leakage_fallback: str = "image"


@dataclass(frozen=True)
class ExportConfig:
    copy_mode: str = "copy"
    overwrite: bool = False
    drop_unknown: bool = True
    min_images_per_class: int = 1
    stats_path: Path = Path("data/metadata/class_stats.csv")


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file_level: str = "DEBUG"
    rotation: str = "10 MB"
    retention: int = 14
    colorize: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Fully resolved configuration (absolute paths, validated values)."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    naming: NamingConfig = field(default_factory=NamingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    extract_frames: ExtractFramesConfig = field(default_factory=ExtractFramesConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Provenance (never hashed, never compared).
    source_path: Optional[Path] = field(default=None, compare=False)
    root: Path = field(default=Path("."), compare=False)
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    # -- helpers ---------------------------------------------------------- #
    def section(self, name: str) -> Mapping[str, Any]:
        """Raw YAML subtree for ``name`` (used for state parameter hashes)."""
        value = self.raw.get(name, {})
        return value if isinstance(value, Mapping) else {}

    def section_hash(self, name: str) -> str:
        return hash_mapping(self.section(name))

    def ensure_dirs(self) -> None:
        for directory in self.paths.output_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        self.paths.state_dir().mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        def _plain(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Mapping):
                return {k: _plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_plain(v) for v in value]
            return value

        data = dataclasses.asdict(self)
        data.pop("raw", None)
        return {k: _plain(v) for k, v in data.items()}


# --------------------------------------------------------------------------- #
# Generic dataclass builder
# --------------------------------------------------------------------------- #


def _coerce(tp: Any, value: Any, path: str) -> Any:
    if value is None:
        return None

    origin = get_origin(tp)

    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]  # noqa: E721
        for arg in args:
            try:
                return _coerce(arg, value, path)
            except (ConfigError, TypeError, ValueError):
                continue
        raise ConfigError(f"{path}: {value!r} does not match any of {args}")

    if origin in (list, List):
        inner = get_args(tp)[0] if get_args(tp) else Any
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return [_coerce(inner, item, f"{path}[{i}]") for i, item in enumerate(value)]

    if origin in (tuple, Tuple):
        args = get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], item, f"{path}[{i}]") for i, item in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))

    if origin in (dict, Dict):
        key_t, val_t = get_args(tp) if get_args(tp) else (Any, Any)
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return {
            _coerce(key_t, k, path): _coerce(val_t, v, f"{path}.{k}")
            for k, v in value.items()
        }

    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)

    if tp is Any:
        return value
    if tp is Path:
        if isinstance(value, Path):
            return value
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a path string, got {type(value).__name__}")
        return Path(value)
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return int(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def _build(cls: Any, data: Any, path: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {unknown}. Valid keys here: {sorted(known)}"
        )
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], f"{path}.{f.name}")
    try:
        return cls(**kwargs)
    except TypeError as exc:  # missing required field
        raise ConfigError(f"{path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins)."""
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(current, value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return dict(data)


def _load_with_base(path: Path, seen: Sequence[Path] = ()) -> Dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(p) for p in (*seen, resolved))
        raise ConfigError(f"circular `base:` chain: {chain}")
    data = _read_yaml(resolved)
    base_ref = data.pop("base", None)
    if base_ref is None:
        return data
    if not isinstance(base_ref, str):
        raise ConfigError(f"{resolved}: `base` must be a string path")
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    return deep_merge(_load_with_base(base_path, (*seen, resolved)), data)


def _parse_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(raw: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` CLI overrides in place (values parsed as YAML)."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        dotted, _, text = item.partition("=")
        keys = [k for k in dotted.strip().split(".") if k]
        if not keys:
            raise ConfigError(f"--set expects a dotted key, got {item!r}")
        cursor: Dict[str, Any] = raw
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = _parse_scalar(text)
    return raw


def hash_mapping(mapping: Mapping[str, Any], length: int = 16) -> str:
    """Stable hash of a (nested) mapping, used for state invalidation."""
    payload = json.dumps(mapping, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()[:length]


#: Sub-directories of ``paths.work_dir`` (key -> directory name).
_WORK_DIR_CHILDREN: Mapping[str, str] = {
    "frames_dir": "frames",
    "crops_dir": "crops",
    "rejected_dir": "rejected",
    "dataset_dir": "dataset",
    "metadata_dir": "metadata",
    "clusters_dir": "clusters",
}


def derive_work_dir(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Expand ``paths.work_dir`` into the artefact directories below it.

    Rule: **a path that is written out explicitly wins; everything else follows
    ``work_dir``.** So one override relocates the whole work tree::

        acdb run-all -c configs/demo.yaml --set paths.work_dir=data/my_show

    puts frames/crops/dataset/metadata/clusters under ``data/my_show/`` (and
    points ``export.stats_path`` at its metadata directory), while
    ``paths.logs_dir`` deliberately stays outside: logs are not a dataset
    artefact. Configurations that list every directory explicitly - as
    ``configs/default.yaml`` does for reference - are left untouched.
    """
    paths = raw.get("paths")
    if not isinstance(paths, Mapping) or "work_dir" not in paths:
        return dict(raw)

    base = Path(str(paths["work_dir"]))
    # POSIX separators keep the raw mapping identical across platforms, which
    # matters because it feeds the state signature hash.
    derived = {
        key: (base / name).as_posix()
        for key, name in _WORK_DIR_CHILDREN.items()
        if key not in paths
    }
    if not derived:
        return dict(raw)

    out: Dict[str, Any] = dict(raw)
    out["paths"] = {**paths, **derived}

    export = dict(out.get("export") or {})
    if "stats_path" not in export:
        metadata_dir = out["paths"].get("metadata_dir")
        if metadata_dir:
            export["stats_path"] = (Path(str(metadata_dir)) / "class_stats.csv").as_posix()
    if export:
        out["export"] = export
    return out


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

_VALID_STRATEGIES = {"interval", "scene", "keyframe"}
_VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}
_VALID_REDUCE = {"pca", "none", "null"}
_VALID_COPY = {"copy", "hardlink", "symlink"}
_VALID_KEEP = {"best_quality", "first", "largest"}
_VALID_PAD = {"edge", "reflect", "constant"}
_VALID_LEAKAGE = {"image", "train_only"}


def _validate(cfg: AppConfig) -> AppConfig:
    problems: List[str] = []

    strategies = tuple(s.lower() for s in cfg.extract_frames.strategies)
    bad = sorted(set(strategies) - _VALID_STRATEGIES)
    if bad:
        problems.append(f"extract_frames.strategies: unknown {bad}; use {sorted(_VALID_STRATEGIES)}")
    if not strategies:
        problems.append("extract_frames.strategies: must not be empty")

    if cfg.project.device not in _VALID_DEVICES and not cfg.project.device.startswith("cuda:"):
        problems.append(
            f"project.device: {cfg.project.device!r} is not one of {sorted(_VALID_DEVICES)} or cuda:N"
        )

    if cfg.extract_frames.interval_seconds <= 0 and cfg.extract_frames.interval_frames is None:
        problems.append("extract_frames.interval_seconds must be > 0 when interval_frames is null")

    lo, hi = cfg.detect.filter.aspect_ratio
    if not 0 < lo < hi:
        problems.append(f"detect.filter.aspect_ratio must satisfy 0 < lo < hi, got {(lo, hi)}")

    if cfg.crop.expand_ratio < 1.0:
        problems.append(f"crop.expand_ratio must be >= 1.0, got {cfg.crop.expand_ratio}")
    if cfg.crop.pad_mode not in _VALID_PAD:
        problems.append(f"crop.pad_mode: {cfg.crop.pad_mode!r} is not one of {sorted(_VALID_PAD)}")

    max_bits = cfg.dedup.phash_size * cfg.dedup.phash_size
    if not 0 <= cfg.dedup.phash_hamming <= max_bits:
        problems.append(
            f"dedup.phash_hamming must be within 0..{max_bits} for phash_size={cfg.dedup.phash_size}"
        )
    if not -1.0 <= cfg.dedup.cosine_threshold <= 1.0:
        problems.append("dedup.cosine_threshold must be within [-1, 1]")
    if cfg.dedup.keep not in _VALID_KEEP:
        problems.append(f"dedup.keep: {cfg.dedup.keep!r} is not one of {sorted(_VALID_KEEP)}")

    if cfg.cluster.reduce_dim.lower() not in _VALID_REDUCE:
        problems.append(f"cluster.reduce_dim: {cfg.cluster.reduce_dim!r} is not one of {sorted(_VALID_REDUCE)}")
    if cfg.cluster.pca_components < 2:
        problems.append("cluster.pca_components must be >= 2")
    if cfg.cluster.representatives_per_cluster < 1:
        problems.append("cluster.representatives_per_cluster must be >= 1")

    ratios = cfg.split.ratios.as_dict()
    if any(v < 0 for v in ratios.values()):
        problems.append(f"split.ratios must be non-negative, got {ratios}")
    total = sum(ratios.values())
    if total <= 0:
        problems.append(f"split.ratios must sum to a positive number, got {ratios}")
    elif abs(total - 1.0) > 1e-6:
        problems.append(f"split.ratios must sum to 1.0, got {total:.6f} ({ratios})")
    if sum(1 for v in ratios.values() if v > 0) < 1:
        problems.append("split.ratios: at least one split must be non-empty")

    weights = dataclasses.asdict(cfg.quality.weights)
    if any(v < 0 for v in weights.values()):
        problems.append(f"quality.weights must be non-negative, got {weights}")
    if sum(weights.values()) <= 0:
        problems.append("quality.weights must not all be zero")

    if cfg.quality.min_score < 0 or cfg.quality.min_score > 1:
        problems.append("quality.min_score must be within [0, 1]")
    if cfg.quality.max_overexposed_ratio > 1 or cfg.quality.max_underexposed_ratio > 1:
        problems.append("quality exposure ratios must be within [0, 1]")

    if cfg.export.copy_mode not in _VALID_COPY:
        problems.append(f"export.copy_mode: {cfg.export.copy_mode!r} is not one of {sorted(_VALID_COPY)}")

    if cfg.split.leakage_fallback not in _VALID_LEAKAGE:
        problems.append(
            f"split.leakage_fallback: {cfg.split.leakage_fallback!r} is not one of "
            f"{sorted(_VALID_LEAKAGE)}"
        )

    if problems:
        joined = "\n  - ".join(problems)
        raise ConfigError(f"invalid configuration ({cfg.source_path}):\n  - {joined}")
    return cfg


def _resolve_paths(cfg: AppConfig, root: Path) -> AppConfig:
    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else (root / p).resolve()

    paths = dataclasses.replace(
        cfg.paths,
        input_dir=_abs(cfg.paths.input_dir),
        work_dir=_abs(cfg.paths.work_dir),
        frames_dir=_abs(cfg.paths.frames_dir),
        crops_dir=_abs(cfg.paths.crops_dir),
        rejected_dir=_abs(cfg.paths.rejected_dir),
        dataset_dir=_abs(cfg.paths.dataset_dir),
        metadata_dir=_abs(cfg.paths.metadata_dir),
        clusters_dir=_abs(cfg.paths.clusters_dir),
        logs_dir=_abs(cfg.paths.logs_dir),
    )
    export = dataclasses.replace(cfg.export, stats_path=_abs(cfg.export.stats_path))
    return dataclasses.replace(cfg, paths=paths, export=export)


def load_config(
    path: Union[str, Path, None] = None,
    *,
    root: Union[str, Path, None] = None,
    overrides: Sequence[str] = (),
) -> AppConfig:
    """Load, merge, override, validate and resolve a configuration file."""
    config_path = Path(path) if path is not None else Path("configs/default.yaml")
    project_root = Path(root).resolve() if root is not None else Path.cwd().resolve()

    raw = _load_with_base(config_path)
    raw = apply_overrides(raw, overrides)
    raw = derive_work_dir(raw)

    cfg: AppConfig = _build(AppConfig, raw, "config")
    cfg = dataclasses.replace(cfg, source_path=config_path.resolve(), root=project_root, raw=raw)
    cfg = _validate(cfg)
    return _resolve_paths(cfg, project_root)
