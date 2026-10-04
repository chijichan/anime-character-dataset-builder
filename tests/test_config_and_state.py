"""Configuration loading, validation, hashing and stage-state bookkeeping."""

from __future__ import annotations

from pathlib import Path

import pytest

from acdb.config import ConfigError, apply_overrides, deep_merge, hash_mapping, load_config
from acdb.utils.state import STAGE_ORDER, StageState, StateStore, canonical_stage


def test_default_and_demo_configs_load(configs_dir: Path) -> None:
    default = load_config(configs_dir / "default.yaml", root=configs_dir.parent)
    demo = load_config(configs_dir / "demo.yaml", root=configs_dir.parent)
    assert default.detect.type == "yolo_v8"
    assert demo.detect.type == "saliency"          # demo overrides the detector
    assert default.extract_frames.strategies == ("interval", "scene")
    assert demo.extract_frames.strategies == ("interval",)  # inherited, then replaced
    assert demo.paths.input_dir.is_absolute()
    assert demo.naming.final_template == default.naming.final_template


def test_base_inheritance_is_recursive(configs_dir: Path) -> None:
    """demo.yaml inherits default.yaml; a nested key keeps its default value."""
    demo = load_config(configs_dir / "demo.yaml", root=configs_dir.parent)
    default = load_config(configs_dir / "default.yaml", root=configs_dir.parent)
    assert demo.crop.expand_ratio == 1.25          # overridden
    assert demo.crop.pad_mode == default.crop.pad_mode  # inherited
    assert demo.video.extensions == default.video.extensions


def test_unknown_key_is_an_error(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("detect:\n  type: saliency\n  filtter: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(config, root=tmp_path)


def test_invalid_values_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        "split:\n  ratios: {train: 0.5, val: 0.1, test: 0.1}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(config, root=tmp_path)


def test_bad_strategy_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("extract_frames:\n  strategies: [magic]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="strategies"):
        load_config(config, root=tmp_path)


def test_cli_overrides_win_and_are_typed(tmp_path: Path) -> None:
    config = tmp_path / "c.yaml"
    config.write_text("detect: {}\n", encoding="utf-8")
    cfg = load_config(
        config,
        root=tmp_path,
        overrides=[
            "detect.params.conf_threshold=0.5",
            "detect.type=saliency",
            "paths.input_dir=videos",
        ],
    )
    assert cfg.detect.type == "saliency"
    assert cfg.detect.params["conf_threshold"] == 0.5
    assert cfg.paths.input_dir == (tmp_path / "videos").resolve()


def test_apply_overrides_creates_missing_branches() -> None:
    raw: dict = {}
    apply_overrides(raw, ["a.b.c=1", "a.b.d=true", "x=y"])
    assert raw == {"a": {"b": {"c": 1, "d": True}}, "x": "y"}
    with pytest.raises(ConfigError):
        apply_overrides(raw, ["nope"])


def test_deep_merge_and_hash_mapping() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}
    assert hash_mapping({"a": 1, "b": 2}) == hash_mapping({"b": 2, "a": 1})
    assert hash_mapping({"a": 1}) != hash_mapping({"a": 2})


# --------------------------------------------------------------------------- #
# Stage state
# --------------------------------------------------------------------------- #


def test_state_round_trip_and_freshness(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    assert store.load("detect") is None
    assert store.stale_reason("detect", "hash") == "no state file"

    upstream = store.upstream_signature("detect")
    signature = store.signature_for("hash", upstream, "detect")
    store.save(
        StageState(
            stage="detect",
            status="complete",
            signature=signature,
            config_hash="hash",
            upstream_signature=upstream,
            items=["a", "b"],
        )
    )
    assert store.is_fresh("detect", "hash")
    assert store.stale_reason("detect", "hash") is None
    assert store.processed_items("detect", "hash") == {"a", "b"}

    # A parameter change invalidates the stage ...
    assert not store.is_fresh("detect", "other-hash")
    assert "parameters changed" in (store.stale_reason("detect", "other-hash") or "")
    # ... and its remembered items must not be trusted any more.
    assert store.processed_items("detect", "other-hash") == set()


def test_upstream_signature_chains_and_invalidates_downstream(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    root_signature = store.upstream_signature("crop")

    # Completing an upstream stage changes the downstream signature.
    upstream = store.upstream_signature("detect")
    store.save(
        StageState(
            stage="detect",
            status="complete",
            signature=store.signature_for("h", upstream, "detect"),
            config_hash="h",
            upstream_signature=upstream,
        )
    )
    assert store.upstream_signature("crop") != root_signature

    # And so does re-running the upstream stage with another config hash.
    upstream2 = store.upstream_signature("detect")
    store.save(
        StageState(
            stage="detect",
            status="complete",
            signature=store.signature_for("h2", upstream2, "detect"),
            config_hash="h2",
            upstream_signature=upstream2,
        )
    )
    assert store.upstream_signature("crop") == store.load("detect").signature  # type: ignore[union-attr]


def test_invalidate_downstream_removes_later_states(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    for stage in STAGE_ORDER:
        store.save(StageState(stage=stage, status="complete", signature=stage))
    removed = store.invalidate_downstream("crop")
    assert "dedup" in removed and "export" in removed
    assert store.load("crop") is not None
    assert store.load("extract_frames") is not None
    assert store.load("dedup") is None


def test_resume_point_and_last_complete(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    assert store.resume_point() == "extract_frames"
    for stage in ("extract_frames", "detect"):
        store.save(StageState(stage=stage, status="complete", signature=stage))
    assert store.resume_point() == "crop"
    assert store.last_complete() == "detect"
    store.save(StageState(stage="crop", status="failed", error="boom"))
    assert store.resume_point() == "crop"
    assert "failed" in (store.stale_reason("crop", "x") or "")


def test_interrupted_stage_is_reported(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    store.save(StageState(stage="detect", status="running"))
    assert "interrupted" in (store.stale_reason("detect", "h") or "")


def test_state_schema_mismatch_is_ignored(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    store.save(StageState(stage="detect", status="complete", signature="s"))
    raw = (tmp_path / "state" / "detect.json").read_text(encoding="utf-8")
    (tmp_path / "state" / "detect.json").write_text(
        raw.replace('"schema_version": 1', '"schema_version": 999'), encoding="utf-8"
    )
    assert store.load("detect") is None


def test_canonical_stage_aliases() -> None:
    from acdb.errors import StateError

    assert canonical_stage("extract-frames") == "extract_frames"
    assert canonical_stage("apply-labels") == "apply_labels"
    assert canonical_stage("quality") == "filter"
    with pytest.raises(StateError):
        canonical_stage("nope")


def test_reusable_items_never_resurrects_removed_inputs() -> None:
    """Carry-over must intersect with the *current* input set, not just state."""
    from acdb.pipeline.context import reusable_items

    done = {"a", "b", "c"}
    assert reusable_items(done, ["a", "b"]) == {"a", "b"}   # 'c' disappeared upstream
    assert reusable_items(done, []) == set()
    assert reusable_items(set(), ["a"]) == set()
    assert reusable_items({1, 2}, [2, 3]) == {"2"}
