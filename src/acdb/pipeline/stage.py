"""Stage base class: freshness check, state bookkeeping, timing, summary.

The runner never re-implements bookkeeping; every stage only writes
:meth:`Stage.execute`. The base class guarantees the invariants the spec asks
for:

* a stage whose signature (own params + upstream signatures) is unchanged is
  skipped, unless ``--force``;
* state is marked ``running`` *before* any output is produced, so a crash is
  visible as ``running`` and the next run reprocesses that stage;
* ``items`` records the inputs actually processed, which is what makes
  ``--resume`` skip finished files inside an interrupted stage;
* the summary line always reports processed / filtered / produced / elapsed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import ClassVar, Dict, Sequence

from acdb.config import AppConfig, hash_mapping
from acdb.errors import StageError
from acdb.pipeline.context import StageContext, StageOutcome
from acdb.utils.logging import stage_summary
from acdb.utils.state import STATUS_COMPLETE, STATUS_FAILED, STATUS_RUNNING, StageState, utc_now

__all__ = ["Stage"]


class Stage(ABC):
    """One pipeline stage."""

    #: Canonical stage name (matches :data:`acdb.utils.state.STAGE_ORDER`).
    name: ClassVar[str] = ""
    #: Primary configuration section (its subtree is hashed into the signature).
    section: ClassVar[str] = ""
    #: Extra sections that also affect this stage's output.
    extra_sections: ClassVar[Sequence[str]] = ()
    #: One-line description shown by ``acdb status`` and the runner.
    description: ClassVar[str] = ""

    # -- signature --------------------------------------------------------- #
    @classmethod
    def config_payload(cls, cfg: AppConfig) -> Dict[str, object]:
        """The config subtree(s) that define this stage's behaviour."""
        payload: Dict[str, object] = {
            section: cfg.section(section) for section in (cls.section, *cls.extra_sections)
        }
        # Naming templates change filenames, hence every stage's output.
        payload["naming"] = cfg.section("naming")
        return payload

    @classmethod
    def config_hash(cls, cfg: AppConfig) -> str:
        return hash_mapping(cls.config_payload(cfg))

    # -- execution --------------------------------------------------------- #
    @abstractmethod
    def execute(self, ctx: StageContext) -> StageOutcome:
        """Do the work. Raising means the stage failed and will be re-run."""

    def run(self, ctx: StageContext) -> StageOutcome:
        """Wrapper used by the runner: freshness, state, timing, summary."""
        ctx.stage = self.name
        cfg_hash = self.config_hash(ctx.cfg)
        upstream = ctx.store.upstream_signature(self.name)
        signature = ctx.store.signature_for(cfg_hash, upstream, self.name)

        if not ctx.force and ctx.store.is_fresh(self.name, cfg_hash):
            ctx.info(f"[{self.name}] up to date - skipping (signature {signature[:8]})")
            previous = ctx.store.load(self.name)
            return StageOutcome(
                processed=0,
                filtered=0,
                produced=0,
                extra={
                    "skipped": True,
                    "signature": signature,
                    "previous_produced": (previous.counts or {}).get("produced", 0) if previous else 0,
                },
            )

        reason = ctx.store.stale_reason(self.name, cfg_hash)
        if reason:
            ctx.info(f"[{self.name}] running ({reason})")

        # Read the "already done" set *before* overwriting the state file, and
        # remember it so an interrupted stage can still skip finished files.
        already_done = set() if ctx.force else ctx.store.processed_items(self.name, cfg_hash)
        ctx.resume_items = already_done

        state = StageState(
            stage=self.name,
            status=STATUS_RUNNING,
            signature=signature,
            config_hash=cfg_hash,
            upstream_signature=upstream,
            params=dict(self.config_payload(ctx.cfg)),
            started_at=utc_now(),
            items=sorted(already_done),
        )
        ctx.store.save(state)
        ctx.state = state

        started = perf_counter()
        try:
            outcome = self.execute(ctx)
        except Exception as exc:
            ctx.flush_state()
            state.status = STATUS_FAILED
            state.error = f"{type(exc).__name__}: {exc}"
            state.duration_sec = perf_counter() - started
            state.finished_at = utc_now()
            ctx.store.save(state)
            if isinstance(exc, StageError):
                raise
            raise StageError(f"stage {self.name} failed: {exc}") from exc

        elapsed = perf_counter() - started
        state.status = STATUS_COMPLETE
        state.counts = {
            "processed": outcome.processed,
            "filtered": outcome.filtered,
            "produced": outcome.produced,
        }
        state.items = sorted(set(outcome.items) | already_done)
        state.outputs = list(outcome.outputs)
        state.duration_sec = elapsed
        state.finished_at = utc_now()
        ctx.store.save(state)

        stage_summary(
            self.name,
            processed=outcome.processed,
            filtered=outcome.filtered,
            produced=outcome.produced,
            seconds=elapsed,
            extra={k: v for k, v in outcome.extra.items() if k != "skipped"},
        )
        return outcome
