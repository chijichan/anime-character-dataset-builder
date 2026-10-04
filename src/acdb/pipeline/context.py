"""Shared plumbing handed to every stage.

A :class:`StageContext` bundles the resolved config, the state store and the
table IO helpers so that a stage body reads as pure pipeline logic:

.. code-block:: python

    def execute(self, ctx: StageContext) -> StageOutcome:
        detections = ctx.load("frames")
        ...
        ctx.save("detections", frame)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from acdb.config import AppConfig
from acdb.io import read_table, relpath, table_path, write_table
from acdb.utils.logging import get_logger
from acdb.utils.naming import crop_filename, frame_filename
from acdb.utils.state import StateStore

__all__ = ["StageContext", "StageOutcome", "reusable_items"]


def reusable_items(done: set, current_ids) -> set:
    """Ids whose previous output may be reused: finished *and* still present.

    Carrying over an id that the upstream stage no longer produces would
    resurrect stale rows (a deleted video, a detection that a stricter
    threshold removed), so the intersection is what the per-item stages use.
    """
    if not done:
        return set()
    return {str(value) for value in done} & {str(value) for value in current_ids}


@dataclass
class StageOutcome:
    """What a stage did - feeds the state file and the summary line."""

    processed: int = 0
    filtered: int = 0
    produced: int = 0
    items: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def merge(self, other: StageOutcome) -> StageOutcome:
        self.processed += other.processed
        self.filtered += other.filtered
        self.produced += other.produced
        self.items.extend(other.items)
        self.outputs.extend(other.outputs)
        self.extra.update(other.extra)
        return self


@dataclass
class StageContext:
    """Everything a stage needs, and nothing more."""

    cfg: AppConfig
    store: StateStore
    force: bool = False
    dry_run: bool = False
    stage: str = ""
    #: Live state record, used by :meth:`mark_processed` for crash-safe resume.
    state: Any = None
    #: Inputs the state store says are already done (empty under ``--force``).
    #: Stages that process many files should skip these and carry their rows over.
    resume_items: set = field(default_factory=set)
    #: Persist the processed-item list every N items. Stages whose items are
    #: expensive (e.g. one video) lower this to 1; cheap per-item stages raise it.
    flush_every: int = 25
    _pending: list = field(default_factory=list)

    # -- crash-safe progress ----------------------------------------------- #
    def mark_processed(self, *items: str) -> None:
        """Record items finished so far; flushes to the state file periodically.

        A hard kill (Ctrl-C twice, power loss, ``kill -9``) leaves the state
        file with everything flushed so far and the status still ``running``, so
        the next ``--resume`` picks up exactly where the run died.
        """
        if self.dry_run or not items:
            return
        self._pending.extend(str(item) for item in items)
        if len(self._pending) >= max(1, self.flush_every):
            self.flush_state()

    def flush_state(self) -> None:
        if self.state is None or not self._pending:
            return
        self.state.items.extend(self._pending)
        del self._pending[:]
        self.store.save(self.state)

    # -- logging ----------------------------------------------------------- #
    @property
    def logger(self):
        return get_logger(self.stage or "pipeline")

    # -- tables ------------------------------------------------------------ #
    def table(self, key: str) -> Path:
        return table_path(self.cfg.paths.metadata_dir, key)

    def exists(self, key: str) -> bool:
        return self.table(key).is_file()

    def load(self, key: str, required: Optional[Sequence[str]] = None) -> pd.DataFrame:
        return read_table(self.table(key), required=required, table=key)

    def load_optional(self, key: str) -> Optional[pd.DataFrame]:
        try:
            return self.load(key)
        except Exception:
            return None

    def save(self, key: str, frame: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> Path:
        if self.dry_run:
            return self.table(key)
        return write_table(frame, self.table(key), columns=columns)

    def carry_over(self, key: str, column: str, values: set) -> pd.DataFrame:
        """Rows of a previously written table whose ``column`` is in ``values``.

        Used by the per-item stages to keep the rows of inputs that the state
        says are already done, so ``--resume`` neither redoes work nor loses
        earlier results.
        """
        if not values or not self.exists(key):
            return pd.DataFrame()
        try:
            frame = self.load(key)
        except Exception:
            return pd.DataFrame()
        if column not in frame.columns:
            return pd.DataFrame()
        return frame.loc[frame[column].astype(str).isin({str(v) for v in values})]

    # -- paths ------------------------------------------------------------- #
    def rel(self, path: Path | str) -> str:
        """Path relative to the project root, POSIX separators."""
        return relpath(path, self.cfg.root)

    def rel_abs(self, value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (self.cfg.root / candidate)

    def frame_path(self, video_id: str, frame_idx: int) -> Path:
        name = frame_filename(video_id, frame_idx, self.cfg.naming.frame_template)
        return self.cfg.paths.frames_dir / video_id / name

    def crop_path(self, video_id: str, frame_idx: int, bbox_idx: int) -> Path:
        name = crop_filename(video_id, frame_idx, bbox_idx, self.cfg.naming.crop_template)
        return self.cfg.paths.crops_dir / video_id / name

    # -- logging helpers --------------------------------------------------- #
    def info(self, message: str, **kwargs: Any) -> None:
        self.logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self.logger.debug(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.logger.warning(message, **kwargs)
