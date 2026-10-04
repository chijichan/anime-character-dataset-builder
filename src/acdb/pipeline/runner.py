"""Stage orchestration: ordering, resume, staleness prompts, provenance."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from acdb.config import AppConfig
from acdb.errors import StageError
from acdb.pipeline.context import StageContext, StageOutcome
from acdb.pipeline.stage import Stage
from acdb.pipeline.stages.apply_labels import ApplyLabelsStage
from acdb.pipeline.stages.cluster import ClusterStage
from acdb.pipeline.stages.crop import CropStage
from acdb.pipeline.stages.dedup import DedupStage
from acdb.pipeline.stages.detect import DetectStage
from acdb.pipeline.stages.export import ExportStage
from acdb.pipeline.stages.extract_frames import ExtractFramesStage
from acdb.pipeline.stages.filter import FilterStage
from acdb.pipeline.stages.split import SplitStage
from acdb.utils.atomic import write_text_atomic
from acdb.utils.logging import get_logger
from acdb.utils.state import STAGE_ORDER, StateStore, canonical_stage

__all__ = [
    "STAGE_CLASSES",
    "build_stages",
    "run_pipeline",
    "freshness_report",
    "stale_stages",
    "config_hashes",
]

STAGE_CLASSES: Dict[str, type] = {
    "extract_frames": ExtractFramesStage,
    "detect": DetectStage,
    "crop": CropStage,
    "dedup": DedupStage,
    "filter": FilterStage,
    "cluster": ClusterStage,
    "apply_labels": ApplyLabelsStage,
    "split": SplitStage,
    "export": ExportStage,
}


def build_stages(names: Optional[Sequence[str]] = None) -> List[Stage]:
    """Instantiate stages in canonical order (all of them when ``names`` is None)."""
    if names:
        wanted = [canonical_stage(name) for name in names]
        ordered = [stage for stage in STAGE_ORDER if stage in set(wanted)]
        unknown = [stage for stage in wanted if stage not in STAGE_CLASSES]
        if unknown:  # pragma: no cover - canonical_stage already validates
            raise StageError(f"no implementation for stage(s) {unknown}")
    else:
        ordered = list(STAGE_ORDER)
    return [STAGE_CLASSES[name]() for name in ordered]


def stage_config_hashes(stages: Sequence[Stage], cfg: AppConfig) -> Dict[str, str]:
    return {stage.name: stage.config_hash(cfg) for stage in stages}


def config_hashes(cfg: AppConfig) -> Dict[str, str]:
    """Config hash of every stage, used for freshness/resume decisions."""
    return stage_config_hashes(build_stages(), cfg)


def freshness_report(cfg: AppConfig, stages: Sequence[Stage]) -> List[dict]:
    """Per-stage freshness, for ``acdb status`` and the pre-run prompt."""
    store = StateStore(cfg.paths.state_dir())
    rows: List[dict] = []
    for stage in stages:
        config_hash = stage.config_hash(cfg)
        state = store.load(stage.name)
        rows.append(
            {
                "stage": stage.name,
                "status": state.status if state else "absent",
                "fresh": store.is_fresh(stage.name, config_hash),
                "reason": store.stale_reason(stage.name, config_hash),
                "items": len(state.items) if state else 0,
                "seconds": round(state.duration_sec, 2) if state else 0.0,
                "signature": (state.signature[:8] if state else ""),
            }
        )
    return rows


def stale_stages(cfg: AppConfig, stages: Sequence[Stage]) -> Dict[str, Optional[str]]:
    store = StateStore(cfg.paths.state_dir())
    return {
        stage.name: store.stale_reason(stage.name, stage.config_hash(cfg)) for stage in stages
    }


def _confirm(question: str, *, assume_yes: bool) -> bool:
    """Ask before re-running; non-interactive sessions default to *yes*."""
    if assume_yes:
        return True
    interactive = False
    try:  # pragma: no cover - depends on the terminal
        interactive = sys.stdin is not None and sys.stdin.isatty()
    except Exception:  # pragma: no cover
        interactive = False
    if not interactive:  # pragma: no cover - CI path
        get_logger("runner").info(f"{question} -> yes (non-interactive session)")
        return True
    import typer  # pragma: no cover - interactive path

    return bool(typer.confirm(question, default=True))  # pragma: no cover


def write_resolved_config(cfg: AppConfig) -> Path:
    """Persist the resolved configuration for reproducibility."""
    import yaml

    target = cfg.paths.metadata_dir / "config_resolved.yaml"
    payload = {
        "source_config": str(cfg.source_path) if cfg.source_path else "",
        "project_root": str(cfg.root),
        "resolved": cfg.to_dict(),
    }
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    return write_text_atomic(target, text)


def _warn_on_config_switch(cfg: AppConfig) -> None:
    """Warn when a *different effective configuration* reuses an output directory.

    Two runs pointing at the same ``paths.metadata_dir`` overwrite each other's
    tables and state. The comparison is on the **resolved** configuration, not on
    the file name, so moving the same settings into a dedicated config file (or
    running the same values from the CLI) stays silent - only a real change
    speaks up, and it names the sections that differ.
    """
    import yaml

    resolved = cfg.paths.metadata_dir / "config_resolved.yaml"
    if not resolved.is_file():
        return
    try:
        previous = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):  # pragma: no cover - defensive
        return
    before = previous.get("resolved")
    if not isinstance(before, Mapping):
        return

    current = cfg.to_dict()
    changed = sorted(
        key
        for key in set(before) | set(current)
        if key not in {"source_path", "root"} and before.get(key) != current.get(key)
    )
    if not changed:
        return
    get_logger("runner").warning(
        f"{cfg.paths.metadata_dir} was last used with a different configuration "
        f"(changed sections: {', '.join(changed)}).\n"
        f"  This run will overwrite the artefacts in that directory. To keep them apart,\n"
        f"  give this run its own tree:  --set paths.work_dir=data/<name>"
    )


def run_pipeline(
    cfg: AppConfig,
    *,
    stages: Optional[Sequence[str]] = None,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    assume_yes: bool = False,
    pause_for_labels: bool = True,
) -> List[StageOutcome]:
    """Run the requested stages (or the rest of the pipeline under ``--resume``).

    ``pause_for_labels`` stops the run just before ``apply-labels`` when not a
    single cluster has been labelled yet, instead of failing later in ``export``
    with an empty dataset. Pass ``False`` to run straight through (useful for
    tests and for exporting an all-``unknown`` dataset on purpose).
    """
    logger = get_logger("runner")
    cfg.ensure_dirs()
    store = StateStore(cfg.paths.state_dir())
    if not dry_run:
        _warn_on_config_switch(cfg)
        write_resolved_config(cfg)

    selected = build_stages(stages)

    if resume:
        resume_point = store.resume_point({s.name: s.config_hash(cfg) for s in selected})
        if resume_point is None:
            logger.info("nothing to resume: every stage is complete and up to date")
            return []
        index = list(STAGE_ORDER).index(resume_point)
        selected = [stage for stage in selected if list(STAGE_ORDER).index(stage.name) >= index]
        logger.info(f"resuming from stage '{resume_point}'")

    if not selected:
        logger.info("no stages selected")
        return []

    # Warn (and ask) when a *parameter* change invalidates completed work. An
    # interrupted or missing stage is normal progress, not something to confirm.
    needs_confirmation = ("parameters changed", "signature does not match")
    changed = {
        stage.name: reason
        for stage in selected
        if (reason := store.stale_reason(stage.name, stage.config_hash(cfg)))
        and any(token in reason for token in needs_confirmation)
        and not force
    }
    if changed:
        for name, reason in changed.items():
            logger.warning(f"[{name}] {reason}")
        downstream = sorted(
            set(changed)
            | {
                stage
                for name in changed
                for stage in STAGE_ORDER[list(STAGE_ORDER).index(name) + 1 :]
            }
        )
        question = (
            f"Re-run {len(downstream)} stage(s) whose parameters changed "
            f"({', '.join(downstream)})?"
        )
        if not _confirm(question, assume_yes=assume_yes):
            raise StageError(
                "aborted: parameters changed but the re-run was declined. "
                "Re-run with --force, or restore the previous configuration."
            )

    outcomes: List[StageOutcome] = []
    for stage in selected:
        if stage.name == "apply_labels" and pause_for_labels and not _has_labels(cfg):
            logger.warning(_labelling_banner(cfg))
            break
        ctx = StageContext(cfg=cfg, store=store, force=force, dry_run=dry_run, stage=stage.name)
        logger.info(f"--- {stage.name}: {stage.description} ---")
        outcomes.append(stage.run(ctx))
    return outcomes


def _has_labels(cfg: AppConfig) -> bool:
    """True when at least one cluster has a real ``anime__character`` label."""
    from acdb.io import read_labels

    try:
        labels = read_labels(cfg.paths.clusters_dir, log=False)
    except Exception:  # pragma: no cover - malformed label file surfaces later
        return True
    return any(not label.is_unknown for label in labels.values())


def _labelling_banner(cfg: AppConfig) -> str:
    from acdb.io import read_labels

    try:
        count = len(read_labels(cfg.paths.clusters_dir, log=False))
    except Exception:  # pragma: no cover
        count = 0
    review = cfg.paths.clusters_dir
    return (
        "\n" + "=" * 72 + "\n"
        f"PAUSED for the human labelling step ({count} cluster(s) waiting in {review})\n"
        "  1. open data/clusters/cluster_labels.yaml and replace `unknown` with\n"
        "     `{anime}__{character}` for each cluster, or rename the review\n"
        "     folders keeping the `cluster_<id>` token (e.g. cluster_003__show__hero);\n"
        "  2. re-run with --resume to apply the labels and export the dataset.\n"
        "Unlabelled clusters stay unknown/ and are dropped when export.drop_unknown\n"
        "is true.\n" + "=" * 72
    )
