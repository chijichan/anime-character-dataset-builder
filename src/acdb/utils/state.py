"""Per-stage bookkeeping under ``data/metadata/.state/{stage}.json``.

Each file records

* ``signature`` - a hash over the stage's own parameters *and* the whole
  upstream signature chain, so any parameter change automatically invalidates
  this stage and everything after it;
* ``items`` - the list of input keys already processed, which is what makes
  ``--resume`` skip work instead of redoing it;
* ``counts``/timings - what :func:`acdb.utils.logging.stage_summary` reports.

State files are written atomically (temp file + rename), so a crash during
bookkeeping cannot corrupt them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from acdb import SCHEMA_VERSION
from acdb.config import hash_mapping
from acdb.errors import StateError
from acdb.utils.atomic import read_json, write_json_atomic

__all__ = ["StageState", "StateStore", "STAGE_ORDER"]

#: Canonical execution order. Used for upstream chaining, ``--resume`` and
#: downstream invalidation.
STAGE_ORDER: Sequence[str] = (
    "extract_frames",
    "detect",
    "crop",
    "dedup",
    "filter",
    "cluster",
    "apply_labels",
    "split",
    "export",
)

#: CLI aliases -> canonical stage names.
STAGE_ALIASES: Mapping[str, str] = {
    "extract-frames": "extract_frames",
    "extract_frames": "extract_frames",
    "frames": "extract_frames",
    "detect": "detect",
    "crop": "crop",
    "dedup": "dedup",
    "filter": "filter",
    "quality": "filter",
    "cluster": "cluster",
    "apply-labels": "apply_labels",
    "apply_labels": "apply_labels",
    "labels": "apply_labels",
    "split": "split",
    "export": "export",
}

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"


def canonical_stage(name: str) -> str:
    try:
        return STAGE_ALIASES[name]
    except KeyError as exc:
        raise StateError(
            f"unknown stage {name!r}; expected one of {sorted(set(STAGE_ALIASES))}"
        ) from exc


@dataclass
class StageState:
    """Persisted state of one stage."""

    stage: str
    status: str = STATUS_PENDING
    signature: str = ""
    config_hash: str = ""
    upstream_signature: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    items: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_sec: float = 0.0
    error: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    # -- (de)serialisation ------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "signature": self.signature,
            "config_hash": self.config_hash,
            "upstream_signature": self.upstream_signature,
            "params": self.params,
            "counts": self.counts,
            "items": self.items,
            "outputs": self.outputs,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": round(self.duration_sec, 3),
            "error": self.error,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StageState:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in payload.items() if k in known}
        state = cls(**data)  # type: ignore[arg-type]
        state.items = list(state.items or [])
        state.outputs = list(state.outputs or [])
        state.counts = dict(state.counts or {})
        state.params = dict(state.params or {})
        return state

    @property
    def is_complete(self) -> bool:
        return self.status == STATUS_COMPLETE

    def processed(self) -> Set[str]:
        return set(self.items)


class StateStore:
    """Read/write access to ``.state/`` plus signature chaining."""

    def __init__(self, state_dir: Path, *, schema_version: int = SCHEMA_VERSION) -> None:
        self.state_dir = Path(state_dir)
        self.schema_version = schema_version
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------- #
    def path_for(self, stage: str) -> Path:
        return self.state_dir / f"{canonical_stage(stage)}.json"

    # -- io ---------------------------------------------------------------- #
    def load(self, stage: str) -> Optional[StageState]:
        stage = canonical_stage(stage)
        payload = read_json(self.path_for(stage))
        if not isinstance(payload, Mapping):
            return None
        if int(payload.get("schema_version", -1)) != self.schema_version:
            # Incompatible bookkeeping: ignore it so the stage re-runs.
            return None
        return StageState.from_dict(payload)

    def save(self, state: StageState) -> Path:
        state.schema_version = self.schema_version
        return write_json_atomic(self.path_for(state.stage), state.to_dict())

    def delete(self, stage: str) -> bool:
        path = self.path_for(stage)
        if path.is_file():
            path.unlink()
            return True
        return False

    # -- signatures -------------------------------------------------------- #
    @staticmethod
    def signature_for(config_hash: str, upstream_signature: str, stage: str) -> str:
        return hash_mapping(
            {
                "stage": stage,
                "config": config_hash,
                "upstream": upstream_signature,
                "schema": SCHEMA_VERSION,
            }
        )

    def upstream_signature(self, stage: str) -> str:
        """Fold every upstream stage's stored signature into one hash."""
        stage = canonical_stage(stage)
        signature = "root"
        for candidate in STAGE_ORDER:
            if candidate == stage:
                break
            state = self.load(candidate)
            if state is None or not state.is_complete:
                marker = state.status if state is not None else "absent"
                signature = hash_mapping({"prev": signature, "stage": candidate, "state": marker})
            else:
                signature = state.signature or hash_mapping(
                    {"prev": signature, "stage": candidate, "state": "complete-no-signature"}
                )
        return signature

    def is_fresh(self, stage: str, config_hash: str) -> bool:
        """True when the stored state matches the current config *and* upstream."""
        stage = canonical_stage(stage)
        state = self.load(stage)
        if state is None or not state.is_complete:
            return False
        expected = self.signature_for(config_hash, self.upstream_signature(stage), stage)
        return state.signature == expected

    def stale_reason(self, stage: str, config_hash: str) -> Optional[str]:
        """Explain why a stage is not fresh (``None`` when it is fresh)."""
        stage = canonical_stage(stage)
        state = self.load(stage)
        if state is None:
            return "no state file"
        if state.status == STATUS_RUNNING:
            return "previous run was interrupted mid-stage"
        if state.status == STATUS_FAILED:
            return f"previous run failed: {state.error or 'unknown error'}"
        if not state.is_complete:
            return f"status is {state.status}"
        upstream = self.upstream_signature(stage)
        if state.upstream_signature != upstream:
            return "an upstream stage changed"
        if state.config_hash != config_hash:
            return f"parameters changed (config hash {state.config_hash[:8]} -> {config_hash[:8]})"
        if state.signature != self.signature_for(config_hash, upstream, stage):
            return "stored signature does not match"
        return None

    # -- resume helpers ---------------------------------------------------- #
    def processed_items(self, stage: str, config_hash: str) -> Set[str]:
        """Items safe to skip.

        Skipping is allowed when the stage completed *or* was interrupted, as
        long as the parameters are unchanged - that is what makes ``--resume``
        pick up a half-finished stage file by file instead of redoing it. A
        different parameter hash invalidates the remembered work completely,
        because those items were produced under different rules.
        """
        state = self.load(stage)
        if state is None or not state.items:
            return set()
        if state.config_hash != config_hash:
            return set()
        return state.processed()

    def resume_point(self, configs: Optional[Mapping[str, str]] = None) -> Optional[str]:
        """First stage that is missing, incomplete, or stale.

        Pass ``configs`` (stage -> config hash) to also treat a *completed but
        invalidated* stage as work to do - that is what makes ``--resume``
        re-run the pipeline after a parameter change instead of reporting it as
        finished.
        """
        for stage in STAGE_ORDER:
            state = self.load(stage)
            if state is None or not state.is_complete:
                return stage
            expected = None if configs is None else configs.get(stage)
            if expected is not None and not self.is_fresh(stage, expected):
                return stage
        return None

    def last_complete(self) -> Optional[str]:
        last: Optional[str] = None
        for stage in STAGE_ORDER:
            state = self.load(stage)
            if state is not None and state.is_complete:
                last = stage
            else:
                break
        return last

    def stale_stages(self, configs: Mapping[str, str]) -> List[str]:
        """Stages whose stored state does not match ``configs`` (stage -> hash)."""
        return [stage for stage in STAGE_ORDER if self.stale_reason(stage, configs[stage])]

    def invalidate_downstream(self, stage: str, *, include_self: bool = False) -> List[str]:
        """Delete state files of every stage after ``stage`` (and optionally itself)."""
        stage = canonical_stage(stage)
        index = list(STAGE_ORDER).index(stage)
        removed: List[str] = []
        targets: Iterable[str] = STAGE_ORDER[index:] if include_self else STAGE_ORDER[index + 1:]
        for candidate in targets:
            if self.delete(candidate):
                removed.append(candidate)
        return removed

    def restart(self, stage: str) -> None:
        """Mark ``stage`` as interrupted so the next run reprocesses it."""
        state = self.load(stage)
        if state is not None:
            state.status = STATUS_PENDING
            state.error = None
            self.save(state)

    def status_table(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for stage in STAGE_ORDER:
            state = self.load(stage)
            if state is None:
                rows.append({"stage": stage, "status": "absent", "items": 0, "seconds": 0.0})
                continue
            rows.append(
                {
                    "stage": stage,
                    "status": state.status,
                    "items": len(state.items),
                    "seconds": round(state.duration_sec, 2),
                    "finished_at": state.finished_at or "",
                    "error": state.error or "",
                }
            )
        return rows


def utc_now() -> str:
    return datetime.now().isoformat(timespec="seconds")
