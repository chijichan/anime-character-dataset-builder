"""``acdb`` - the command line interface.

One subcommand per pipeline stage, plus three conveniences::

    acdb run-all            run (or resume) the whole pipeline
    acdb status             show which stages are complete / stale and why
    acdb demo-video         render the synthetic demo footage
    acdb demo-annotate      simulate the human labelling step (demo only)
    acdb init-config        copy a config to start from

Every stage command accepts the same knobs::

    -c/--config PATH        config file (default: configs/default.yaml)
    --root PATH             project root for relative paths (default: cwd)
    --set a.b=c              override any config value (repeatable)
    --force                 re-run even if the stage is up to date
    --dry-run               do everything but write files
    -y/--yes                assume "yes" for the re-run confirmation
    -v/--verbose            verbose (DEBUG) console logging
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, List, NoReturn, Optional, Sequence

import typer

from acdb import __version__
from acdb.config import AppConfig, load_config
from acdb.errors import AcbdError, ConfigError
from acdb.pipeline.runner import build_stages, config_hashes, freshness_report, run_pipeline
from acdb.pipeline.stages import STAGE_CLASSES
from acdb.utils.logging import get_logger, setup_logging
from acdb.utils.state import STAGE_ORDER, StateStore

__all__ = ["app", "main"]

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Build an anime character classification dataset from raw video.",
)

ConfigOpt = Annotated[
    Optional[Path], typer.Option("--config", "-c", help="Path to the YAML configuration.")
]
RootOpt = Annotated[
    Optional[Path], typer.Option("--root", help="Project root for relative paths (default: cwd).")
]
SetOpt = Annotated[
    Optional[List[str]],
    typer.Option("--set", help="Override a config value, e.g. --set detect.filter.min_conf=0.5"),
]
ForceOpt = Annotated[bool, typer.Option("--force", help="Re-run even when the stage is up to date.")]
DryRunOpt = Annotated[bool, typer.Option("--dry-run", help="Do the work but write nothing.")]
YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="Assume yes for confirmations.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="DEBUG console logging.")]
ResumeOpt = Annotated[bool, typer.Option("--resume", help="Continue from the first unfinished stage.")]

DEFAULT_CONFIG = "configs/default.yaml"


def _fail(message: str) -> NoReturn:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _prepare(
    config: Optional[Path],
    root: Optional[Path],
    overrides: Optional[Sequence[str]],
    verbose: bool,
) -> AppConfig:
    """Load + validate the config and install logging."""
    try:
        cfg = load_config(config or DEFAULT_CONFIG, root=root, overrides=list(overrides or []))
    except ConfigError as exc:
        _fail(str(exc))
    cfg.ensure_dirs()
    setup_logging(
        logs_dir=cfg.paths.logs_dir,
        level="DEBUG" if verbose else cfg.logging.level,
        file_level=cfg.logging.file_level,
        rotation=cfg.logging.rotation,
        retention=cfg.logging.retention,
        colorize=cfg.logging.colorize,
    )
    get_logger("cli").debug(f"config: {cfg.source_path} (root {cfg.root})")
    return cfg


def _report(outcomes: Sequence[object], *, dry_run: bool) -> None:
    executed = [o for o in outcomes if not (getattr(o, "extra", {}) or {}).get("skipped")]
    skipped = [o for o in outcomes if (getattr(o, "extra", {}) or {}).get("skipped")]
    if dry_run:
        typer.secho("dry run: nothing was written", fg=typer.colors.YELLOW)
    typer.secho(
        f"done: {len(executed)} stage(s) executed, {len(skipped)} skipped as up to date",
        fg=typer.colors.GREEN,
    )


def _execute(
    stages: Optional[Sequence[str]],
    config: Optional[Path],
    root: Optional[Path],
    overrides: Optional[Sequence[str]],
    *,
    force: bool,
    dry_run: bool,
    yes: bool,
    verbose: bool,
    resume: bool = False,
) -> None:
    cfg = _prepare(config, root, overrides, verbose)
    try:
        outcomes = run_pipeline(
            cfg,
            stages=stages,
            force=force,
            resume=resume,
            dry_run=dry_run,
            assume_yes=yes,
        )
    except AcbdError as exc:
        get_logger("cli").error(str(exc))
        _fail(str(exc))
    except (KeyboardInterrupt, typer.Abort):  # pragma: no cover - interactive
        typer.secho(
            "\nstopped - nothing was left half-written; re-run with --resume to continue",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=130) from None
    _report(outcomes, dry_run=dry_run)


# --------------------------------------------------------------------------- #
# One command per stage
# --------------------------------------------------------------------------- #


def _stage_command(stage: str, description: str):
    def command(
        config: ConfigOpt = None,
        root: RootOpt = None,
        set_: SetOpt = None,
        force: ForceOpt = False,
        dry_run: DryRunOpt = False,
        yes: YesOpt = False,
        verbose: VerboseOpt = False,
    ) -> None:
        _execute(
            [stage], config, root, set_, force=force, dry_run=dry_run, yes=yes, verbose=verbose
        )

    command.__name__ = stage
    command.__doc__ = description
    return command


for _stage_name, _stage_class in STAGE_CLASSES.items():
    app.command(_stage_name.replace("_", "-"))(
        _stage_command(_stage_name, _stage_class.description)
    )


# --------------------------------------------------------------------------- #
# Whole-pipeline commands
# --------------------------------------------------------------------------- #


@app.command("run-all")
def run_all(
    config: ConfigOpt = None,
    root: RootOpt = None,
    set_: SetOpt = None,
    force: ForceOpt = False,
    resume: ResumeOpt = False,
    dry_run: DryRunOpt = False,
    yes: YesOpt = False,
    verbose: VerboseOpt = False,
    start: Annotated[
        Optional[str], typer.Option("--from", help="Start at this stage instead of the first.")
    ] = None,
    stop: Annotated[
        Optional[str], typer.Option("--to", help="Stop after this stage.")
    ] = None,
) -> None:
    """Run the whole pipeline: extract-frames -> detect -> ... -> export."""
    stages: Optional[List[str]] = None
    if start or stop:
        from acdb.utils.state import canonical_stage

        try:
            first = STAGE_ORDER.index(canonical_stage(start)) if start else 0
            last = STAGE_ORDER.index(canonical_stage(stop)) if stop else len(STAGE_ORDER) - 1
        except (ValueError, AcbdError) as exc:
            _fail(str(exc))
        if first > last:
            _fail(f"--from {start} comes after --to {stop}")
        stages = list(STAGE_ORDER[first : last + 1])
    _execute(
        stages, config, root, set_, force=force, dry_run=dry_run, yes=yes,
        verbose=verbose, resume=resume,
    )


@app.command("status")
def status(
    config: ConfigOpt = None,
    root: RootOpt = None,
    set_: SetOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Show per-stage freshness and the command to run next."""
    cfg = _prepare(config, root, set_, verbose)
    stages = build_stages()
    rows = freshness_report(cfg, stages)
    typer.secho(f"{'stage':<16}{'status':<10}{'items':>7}{'sec':>9}  fresh  note", bold=True)
    for row in rows:
        note = "" if row["fresh"] else (row["reason"] or "")
        typer.echo(
            f"{row['stage']:<16}{row['status']:<10}{row['items']:>7}"
            f"{row['seconds']:>9.2f}  {'yes' if row['fresh'] else 'no ':>5}  {note}"
        )
    next_stage = StateStore(cfg.paths.state_dir()).resume_point(config_hashes(cfg))
    typer.echo("")
    if next_stage is None:
        typer.secho("pipeline complete - nothing to do", fg=typer.colors.GREEN)
    else:
        typer.secho(
            f"next: acdb {next_stage.replace('_', '-')} -c {cfg.source_path}", fg=typer.colors.CYAN
        )
    typer.echo(f"logs: {cfg.paths.logs_dir}")


@app.command("init-config")
def init_config(
    target: Annotated[Path, typer.Argument(help="Where to write the configuration.")] = Path(
        "configs/my.yaml"
    ),
    source: Annotated[
        Optional[Path], typer.Option("--from", help="Template to copy (default: configs/default.yaml).")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Copy a starter configuration (default or demo) to a new path."""
    import shutil

    template = source or (Path(__file__).resolve().parents[2] / "configs" / "default.yaml")
    if not template.is_file():
        template = Path(DEFAULT_CONFIG)
    if not template.is_file():
        _fail(f"no template found at {template}")
    if target.exists() and not force:
        _fail(f"{target} already exists (use --force to overwrite)")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, target)
    typer.secho(f"wrote {target} (based on {template})", fg=typer.colors.GREEN)


@app.command("demo-video")
def demo_video(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory that will hold the synthetic videos.")
    ] = Path("data/input"),
    seconds: Annotated[float, typer.Option("--seconds", help="Duration of each episode.")] = 12.0,
    fps: Annotated[int, typer.Option("--fps", help="Frames per second.")] = 12,
    width: Annotated[int, typer.Option("--width", help="Frame width in pixels.")] = 640,
    height: Annotated[int, typer.Option("--height", help="Frame height in pixels.")] = 360,
    episodes: Annotated[
        int, typer.Option("--episodes", help="Episodes per show (>=3 for a leak-free split).")
    ] = 3,
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing videos.")] = False,
) -> None:
    """Render the synthetic demo footage (no downloads, no copyrighted material)."""
    from acdb.demo import DEMO_CHARACTERS, make_demo_videos

    target = Path(output)
    existing = sorted(target.glob("*/*.mp4")) if target.is_dir() else []
    if existing and not force:
        typer.secho(
            f"{len(existing)} video(s) already in {target}; use --force to regenerate",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=0)
    setup_logging(logs_dir=Path("logs"), level="INFO")
    manifest = make_demo_videos(
        target, width=width, height=height, fps=fps, seconds=seconds, episodes=episodes
    )
    typer.secho(
        f"wrote {len(manifest['videos'])} demo video(s) to {target} "
        f"({len(DEMO_CHARACTERS)} characters across 3 shows)",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"ground truth: {target / 'demo_ground_truth.json'}")


@app.command("demo-annotate")
def demo_annotate(
    config: ConfigOpt = None,
    root: RootOpt = None,
    set_: SetOpt = None,
    truth: Annotated[
        Optional[Path],
        typer.Option("--truth", help="Ground-truth manifest from `acdb demo-video`."),
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Simulate the human labelling step by matching cluster palettes (demo only)."""
    from acdb.demo import simulate_cluster_labels
    from acdb.extractors import build_extractor
    from acdb.utils.device import resolve_device

    cfg = _prepare(config, root, set_, verbose)
    manifest = truth or (cfg.paths.input_dir / "demo_ground_truth.json")
    extractor = build_extractor(cfg.features, resolve_device(cfg.project.device))
    try:
        result = simulate_cluster_labels(cfg.paths.clusters_dir, manifest, extractor)
    except AcbdError as exc:
        _fail(str(exc))
    finally:
        extractor.close()
    typer.secho(f"wrote {result['path']}", fg=typer.colors.GREEN)
    for detail in result.get("details", []):
        extra = detail.get("reason", "")
        if detail.get("similarity") is not None:
            extra = f"similarity={detail['similarity']} (runner-up {detail.get('runner_up', '-')} {detail.get('runner_up_similarity', '-')})"
        typer.echo(f"  cluster {detail['cluster']}: {detail['label']} {extra}")


@app.callback(invoke_without_command=True)
def _root(
    version: Annotated[
        bool, typer.Option("--version", help="Print the version and exit.", is_eager=True)
    ] = False,
) -> None:
    if version:
        typer.echo(f"acdb {__version__}")
        raise typer.Exit(code=0)


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except AcbdError as exc:  # pragma: no cover - safety net
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
