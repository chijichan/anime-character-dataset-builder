"""End-to-end smoke test: a tiny synthetic video through all nine stages.

The test renders two seconds of footage with :mod:`acdb.demo`, runs the real
pipeline with the demo configuration (classical detector + colour-histogram
embeddings, so no weights and no network), then asserts the hard output
contract: the ImageFolder layout, ``metadata.csv`` and ``classes.json``.

It also covers the two operational promises: re-running a stage skips it, and
``--resume`` after an interruption does not reprocess finished work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from acdb.demo import simulate_cluster_labels
from acdb.extractors import build_extractor
from acdb.io import METADATA_COLUMNS, read_classes_json, read_table
from acdb.pipeline.runner import config_hashes, run_pipeline
from acdb.utils.naming import parse_final_filename
from acdb.utils.state import STAGE_ORDER, StateStore

pytestmark = pytest.mark.slow

INPUT_OVERRIDES = [
    # Small frames and few samples: this only has to exercise the code paths.
    "extract_frames.interval_seconds=1.0",
    "extract_frames.max_frames_per_video=5",
    "detect.params.min_area_px=120",
    "detect.filter.min_side=16",
    "detect.filter.min_area_ratio=0.001",
    "crop.min_output_side=16",
    "quality.min_laplacian_var=1.0",
    "quality.min_side_px=16",
    "quality.min_colorfulness=0.0",
    "quality.min_score=0.0",
    "quality.max_overexposed_ratio=1.0",
    "quality.max_underexposed_ratio=1.0",
    "cluster.type=kmeans",
    "cluster.params.n_clusters=2",
    "cluster.reduce_dim=none",
    "split.min_images_per_class=1",
    "export.min_images_per_class=1",
    "export.overwrite=true",
]


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory) -> dict:
    """Render the footage and run the whole pipeline once for this module."""
    from acdb.config import load_config

    root = tmp_path_factory.mktemp("acdb_e2e")
    configs = Path(__file__).resolve().parents[1] / "configs"
    cfg = load_config(
        configs / "demo.yaml",
        root=root,
        overrides=INPUT_OVERRIDES
        + [
            "paths.input_dir=input",
            "paths.work_dir=work",
            "paths.frames_dir=work/frames",
            "paths.crops_dir=work/crops",
            "paths.rejected_dir=work/rejected",
            "paths.dataset_dir=work/dataset",
            "paths.metadata_dir=work/metadata",
            "paths.clusters_dir=work/clusters",
            "paths.logs_dir=work/logs",
            "export.stats_path=work/metadata/class_stats.csv",
        ],
    )
    cfg.ensure_dirs()

    from acdb.demo import make_demo_videos

    try:
        make_demo_videos(cfg.paths.input_dir, width=320, height=180, fps=6, seconds=4.0,
                         episodes=3, shows=["demo_show_a", "demo_show_b"])
    except Exception as exc:  # pragma: no cover - codec dependent
        pytest.skip(f"cannot render the synthetic video here: {exc}")

    # Phase 1: everything up to the review folders. `pause_for_labels=False`
    # lets apply-labels run on an all-unknown label file - the pipeline must
    # survive that (it just exports nothing) rather than crash.
    first = run_pipeline(
        cfg,
        stages=["extract_frames", "detect", "crop", "dedup", "filter", "cluster"],
        assume_yes=True,
    )
    # Phase 2: the human-in-the-loop step, simulated from the reference sheets.
    extractor = build_extractor(cfg.features, "cpu")
    try:
        labelling = simulate_cluster_labels(
            cfg.paths.clusters_dir, cfg.paths.input_dir / "demo_ground_truth.json", extractor
        )
    finally:
        extractor.close()
    # Phase 3: fold the labels back in and export.
    second = run_pipeline(cfg, assume_yes=True)
    import json

    truth = json.loads((cfg.paths.input_dir / "demo_ground_truth.json").read_text(encoding="utf-8"))
    characters = sorted(
        f"{anime}__{character}"
        for anime, entry in truth["palette"].items()
        for character in entry
    )
    return {
        "cfg": cfg,
        "outcomes": first + second,
        "labelling": labelling,
        "root": root,
        "ground_truth_characters": characters,
    }


def test_all_stages_completed(demo_run) -> None:
    store = StateStore(demo_run["cfg"].paths.state_dir())
    for stage in STAGE_ORDER:
        state = store.load(stage)
        assert state is not None, f"{stage} never ran"
        assert state.status == "complete", f"{stage}: {state.status} {state.error}"


def test_second_run_is_skipped(demo_run) -> None:
    cfg = demo_run["cfg"]
    second = run_pipeline(cfg, assume_yes=True)
    assert all(outcome.extra.get("skipped") for outcome in second), (
        "an unchanged configuration must skip every stage"
    )


def test_resume_after_invalidation(demo_run) -> None:
    cfg = demo_run["cfg"]
    store = StateStore(cfg.paths.state_dir())
    store.invalidate_downstream("filter")
    assert store.resume_point() == "cluster"
    outcomes = run_pipeline(cfg, resume=True, assume_yes=True)
    assert [outcome.extra.get("skipped", False) for outcome in outcomes] == [False] * len(outcomes)
    assert store.resume_point() is None


def test_imagefolder_layout_and_contract(demo_run) -> None:
    cfg = demo_run["cfg"]
    dataset = cfg.paths.dataset_dir
    metadata_path = cfg.paths.metadata_dir / "metadata.csv"
    classes_path = cfg.paths.metadata_dir / "classes.json"
    assert metadata_path.is_file() and classes_path.is_file()

    table = read_table(metadata_path, table="metadata")
    assert list(table.columns) == list(METADATA_COLUMNS)
    assert len(table) > 0
    assert table["image_id"].is_unique

    for row in table.to_dict("records"):
        image = cfg.root / str(row["file_path"])
        assert image.is_file(), f"metadata points at a missing file: {image}"
        assert image.parent.name == f"{row['anime']}__{row['character']}"
        assert image.parent.parent.name == row["split"]
        parsed = parse_final_filename(image.name)
        assert parsed is not None, f"{image.name} does not follow the naming contract"
        assert parsed["anime"] == row["anime"]
        assert parsed["character"] == row["character"]
        assert parsed["video_id"] == row["video_id"]
        assert parsed["frame_idx"] == int(row["frame_idx"])

    classes = read_classes_json(classes_path)
    assert classes, "classes.json must not be empty"
    assert [entry.id for entry in classes] == list(range(len(classes)))
    for entry in classes:
        assert (dataset / "train" / entry.key).exists() or any(
            (dataset / split / entry.key).exists() for split in ("train", "val", "test")
        )


def test_split_directories_are_not_empty(demo_run) -> None:
    """With per-image fallback splitting every split must contain something."""
    cfg = demo_run["cfg"]
    for split in ("train", "val", "test"):
        assert list((cfg.paths.dataset_dir / split).glob("*/*.jpg")), f"{split} is empty"


def test_review_folders_and_labels(demo_run) -> None:
    cfg = demo_run["cfg"]
    clusters = sorted(cfg.paths.clusters_dir.glob("cluster_*"))
    assert clusters, "cluster stage must export review folders"
    assert (clusters[0] / "index.csv").is_file()
    assert (cfg.paths.clusters_dir / "cluster_labels.yaml").is_file()


def test_per_file_resume_after_interruption(demo_run) -> None:
    """An interrupted stage must not reprocess the files it already finished."""
    cfg = demo_run["cfg"]
    store = StateStore(cfg.paths.state_dir())
    state = store.load("extract_frames")
    assert state is not None and len(state.items) >= 3, "need several videos for this test"

    finished, pending = state.items[0], state.items[1:]

    # Rewind the stage as a hard kill would have left it: status `running`,
    # one video still recorded, and that video's rows still in frames.csv.
    state.status = "running"
    state.items = [finished]
    store.save(state)

    frames_path = cfg.paths.metadata_dir / "frames.csv"
    table = read_table(frames_path, table="frames")
    kept = table.loc[table["video_path"] == finished]
    assert not kept.empty
    from acdb.io import write_table

    write_table(kept, frames_path)

    outcomes = run_pipeline(cfg, stages=["extract_frames"], force=False, assume_yes=True)
    outcome = outcomes[0]
    assert outcome.processed == len(pending), (
        f"only the unfinished video(s) should be processed, got {outcome.processed}"
    )

    rewritten = read_table(frames_path, table="frames")
    assert rewritten["video_path"].nunique() == len(state.items) + len(pending), (
        "the resumed run must keep the already-finished videos' rows (carry-over)"
    )
    assert set(rewritten["video_path"]) == set(table["video_path"])


def test_simulated_annotator_labelled_the_clusters(demo_run) -> None:
    """The demo annotator matches clusters to the rendered reference sheets."""
    labelling = demo_run["labelling"]
    assert labelling["written"]
    labels = {int(key): value for key, value in labelling["labels"].items()}
    assert labels, "at least one cluster must be labelled"
    assert all(value != "unknown" for value in labels.values()), (
        f"the synthetic characters are easy to tell apart: {labels}"
    )
    # Every labelled cluster must name one of the six synthetic characters.
    assert set(labels.values()) <= set(demo_run["ground_truth_characters"])


def test_editing_labels_invalidates_downstream(demo_run) -> None:
    cfg = demo_run["cfg"]
    store = StateStore(cfg.paths.state_dir())
    assert store.resume_point(config_hashes(cfg)) is None, (
        "the pipeline is complete before the edit"
    )

    path = cfg.paths.clusters_dir / "cluster_labels.yaml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text + "# reviewed by hand\n", encoding="utf-8")

    assert store.resume_point(config_hashes(cfg)) == "apply_labels", (
        "editing cluster_labels.yaml must invalidate apply_labels and its downstream"
    )
    outcomes = run_pipeline(cfg, assume_yes=True)
    executed = [outcome for outcome in outcomes if not outcome.extra.get("skipped")]
    assert len(executed) == 3, (
        "only apply_labels, split and export should re-run; "
        f"got {[o.extra.get('produced') for o in outcomes]}"
    )
    assert store.resume_point(config_hashes(cfg)) is None
