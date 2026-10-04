"""Stage 4 - ``dedup``: near-duplicate removal, coarse then fine.

Two gates, exactly as specified:

1. **Coarse (pHash).** Perceptual hashes are unpacked into bit vectors and an
   exact Hamming-distance matrix is computed with BLAS (``bits @ (1-bits).T +
   (1-bits) @ bits.T``), row-chunked so memory stays bounded. Pairs at or below
   ``dedup.phash_hamming`` become *candidates*.
2. **Fine (embedding cosine).** Each candidate pair must also reach
   ``dedup.cosine_threshold`` cosine similarity in feature space. Requiring both
   is what keeps pHash's colour blindness from merging two characters who share
   a silhouette and palette layout.

Verified pairs are merged with union-find; one member per group survives
according to ``dedup.keep``. Groups larger than ``dedup.max_group_size`` are
treated as *over-merged* and left untouched (with a warning) - that safety valve
stops one bad edge from collapsing an entire character into a single image.

Output: ``dedup.csv`` (``crops.csv`` + ``kept`` / ``dedup_group``).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from acdb.errors import StageError
from acdb.extractors import build_extractor, compute_features
from acdb.pipeline.context import StageContext, StageOutcome
from acdb.pipeline.scoring import score_metrics
from acdb.pipeline.stage import Stage
from acdb.utils.device import resolve_device
from acdb.utils.image import imread, quality_metrics
from acdb.utils.logging import get_logger

__all__ = [
    "DedupStage",
    "hashes_to_bits",
    "hamming_matrix",
    "candidate_pairs",
    "greedy_groups",
]


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #


def hashes_to_bits(hashes: Sequence[str]) -> np.ndarray:
    """Unpack hex pHashes into a ``(N, bits)`` uint8 matrix of 0/1."""
    cleaned = [str(value or "").strip().lower() for value in hashes]
    width = max((len(value) for value in cleaned), default=0)
    if width == 0:
        return np.zeros((len(cleaned), 0), dtype=np.uint8)
    if width % 2:
        width += 1
    rows = np.empty((len(cleaned), width * 4), dtype=np.uint8)
    for index, value in enumerate(cleaned):
        try:
            raw = bytes.fromhex(value.zfill(width))
        except ValueError as exc:
            raise StageError(f"invalid perceptual hash {value!r}") from exc
        rows[index] = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return rows


def hamming_matrix(bits: np.ndarray, *, chunk_rows: int = 0, budget: int = 20_000_000) -> np.ndarray:
    """Exact pairwise Hamming distances (uint16) for a ``(N, B)`` bit matrix."""
    matrix = np.asarray(bits, dtype=np.float32)
    count = matrix.shape[0]
    if count == 0:
        return np.zeros((0, 0), dtype=np.uint16)
    complement = 1.0 - matrix
    rows = chunk_rows or max(1, min(count, budget // max(1, count)))
    distances = np.empty((count, count), dtype=np.uint16)
    for start in range(0, count, rows):
        stop = min(count, start + rows)
        block = matrix[start:stop]
        distances[start:stop] = np.rint(
            block @ complement.T + complement[start:stop] @ matrix.T
        ).astype(np.uint16)
    return distances


def candidate_pairs(
    distances: np.ndarray,
    threshold: int,
    *,
    groups: Sequence[str] | None = None,
    same_group_only: bool = False,
) -> List[Tuple[int, int]]:
    """Index pairs ``(i, j)`` with ``i < j`` and distance <= ``threshold``."""
    if distances.size == 0:
        return []
    upper = np.triu(distances <= int(threshold), k=1)
    rows, columns = np.nonzero(upper)
    pairs = list(zip(rows.tolist(), columns.tolist()))
    if same_group_only and groups is not None:
        labels = list(groups)
        pairs = [pair for pair in pairs if labels[pair[0]] == labels[pair[1]]]
    return pairs


def greedy_groups(
    order: Sequence[int],
    neighbours: Mapping[int, Sequence[int]],
    *,
    max_group_size: int = 50,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Group near-duplicates around a representative, never transitively.

    ``order`` is the priority in which representatives are chosen (best first).
    For each not-yet-assigned image the image becomes a group representative and
    absorbs every unassigned neighbour - i.e. an image that is close to *the
    representative*.

    Why not connected components? Two frames of the same character one second
    apart are near-identical, and so are the next two: transitive closure chains
    those into one giant group and throws away genuinely different poses. Anchoring
    every group on a representative bounds the group's diameter by the thresholds
    themselves.

    Returns ``(groups, overfull)``; ``overfull`` lists groups that exceeded
    ``max_group_size`` - their surplus members are *kept* rather than deleted, so
    a pathological merge can never erase a whole class.
    """
    assigned: set = set()
    groups: List[List[int]] = []
    overfull: List[List[int]] = []
    for candidate in order:
        if candidate in assigned:
            continue
        members = [candidate]
        assigned.add(candidate)
        for neighbour in neighbours.get(candidate, ()):
            if neighbour in assigned:
                continue
            if len(members) >= max_group_size:
                overfull.append([candidate, neighbour])
                break
            members.append(neighbour)
            assigned.add(neighbour)
        groups.append(members)
    return [group for group in groups if len(group) > 1], overfull


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #


class DedupStage(Stage):
    name = "dedup"
    section = "dedup"
    extra_sections = ("features",)
    description = "remove near-duplicate crops (pHash coarse + cosine fine)"

    def execute(self, ctx: StageContext) -> StageOutcome:
        cfg = ctx.cfg
        settings = cfg.dedup
        logger = get_logger(self.name)

        crops = ctx.load("crops", required=["image_id", "phash", "crop_path", "video_id"])
        if crops.empty:
            raise StageError("crops.csv is empty - run `acdb crop` first")

        frame = crops.copy()
        frame["kept"] = True
        frame["dedup_group"] = ""

        if len(frame) < 2:
            ctx.info("fewer than 2 crops; nothing to deduplicate")
            if not ctx.dry_run:
                ctx.save("dedup", frame)
            return StageOutcome(
                processed=len(frame), filtered=0, produced=len(frame),
                items=frame["image_id"].astype(str).tolist(),
                outputs=[str(ctx.table("dedup"))],
            )

        bits = hashes_to_bits(frame["phash"].tolist())
        distances = hamming_matrix(bits)
        pairs = candidate_pairs(
            distances,
            int(settings.phash_hamming),
            groups=frame["video_id"].astype(str).tolist(),
            same_group_only=bool(settings.same_video_only),
        )
        ctx.info(
            f"pHash coarse gate: {len(pairs)} candidate pair(s) "
            f"(hamming <= {settings.phash_hamming})"
        )
        if not pairs:
            if not ctx.dry_run:
                ctx.save("dedup", frame)
            return StageOutcome(
                processed=len(frame), filtered=0, produced=len(frame),
                items=frame["image_id"].astype(str).tolist(),
                outputs=[str(ctx.table("dedup"))],
                extra={"candidate_pairs": 0, "duplicates": 0},
            )

        device = resolve_device(cfg.project.device)
        extractor = build_extractor(cfg.features, device)
        feature_sets = compute_features(
            frame["image_id"].astype(str).tolist(),
            loader=lambda image_id: imread(ctx.rel_abs(self._path_of(frame, image_id))),
            extractor=extractor,
            cache_path=ctx.table("features"),
            batch_size=cfg.features.batch_size,
        )
        extractor.close()

        vectors = feature_sets.vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        neighbours: Dict[int, List[int]] = {}
        verified = 0
        for left, right in pairs:
            denominator = float(norms[left] * norms[right])
            if denominator <= 0:
                continue
            similarity = float(np.dot(vectors[left], vectors[right]) / denominator)
            if similarity < settings.cosine_threshold:
                continue
            verified += 1
            neighbours.setdefault(left, []).append(right)
            neighbours.setdefault(right, []).append(left)
        ctx.info(
            f"cosine fine gate: {verified}/{len(pairs)} pair(s) confirmed "
            f"(cosine >= {settings.cosine_threshold})"
        )

        # Representatives are chosen best-quality-first so the survivor of a
        # group is the sharpest, best-exposed frame of that group.
        quality_order = self._quality_order(ctx, frame)
        groups, overfull = greedy_groups(
            quality_order, neighbours, max_group_size=int(settings.max_group_size)
        )
        for members in overfull:
            logger.warning(
                f"duplicate group at {frame['image_id'].iloc[members[0]]} exceeded "
                f"dedup.max_group_size={settings.max_group_size}; surplus member kept"
            )

        duplicates = 0
        for group_index, members in enumerate(sorted(groups)):
            group_id = f"g{group_index:04d}"
            survivor = self._pick_survivor(ctx, frame, members, settings.keep)
            for position in members:
                frame.iat[position, frame.columns.get_loc("dedup_group")] = group_id
                if position == survivor:
                    continue
                frame.iat[position, frame.columns.get_loc("kept")] = False
                duplicates += 1

        kept = int(frame["kept"].sum())
        if not ctx.dry_run:
            ctx.save("dedup", frame)
        ctx.info(
            f"dedup.csv: {len(frame)} row(s), {len(groups)} duplicate group(s), "
            f"{duplicates} dropped, {kept} kept"
        )
        return StageOutcome(
            processed=len(frame),
            filtered=duplicates,
            produced=kept,
            items=frame["image_id"].astype(str).tolist(),
            outputs=[str(ctx.table("dedup"))],
            extra={
                "candidate_pairs": len(pairs),
                "confirmed_pairs": verified,
                "groups": len(groups),
                "oversized_groups": len(overfull),
            },
        )

    # -- helpers ----------------------------------------------------------- #
    def _quality_order(self, ctx: StageContext, frame: pd.DataFrame) -> List[int]:
        """Row positions sorted best-first (score, then confidence, then index)."""
        scores = []
        for record in frame.to_dict("records"):
            try:
                image = imread(ctx.rel_abs(str(record["crop_path"])))
                scores.append(score_metrics(quality_metrics(image), ctx.cfg.quality))
            except StageError:  # pragma: no cover - unreadable crop
                scores.append(-1.0)
        frame["_score"] = scores
        return sorted(
            range(len(frame)),
            key=lambda position: (
                -float(frame["_score"].iloc[position]),
                -float(frame["det_conf"].iloc[position]),
                position,
            ),
        )
    @staticmethod
    def _path_of(frame: pd.DataFrame, image_id: str) -> str:
        match = frame.loc[frame["image_id"].astype(str) == image_id, "crop_path"]
        if match.empty:  # pragma: no cover - defensive
            raise StageError(f"unknown image_id {image_id!r}")
        return str(match.iloc[0])

    def _pick_survivor(
        self,
        ctx: StageContext,
        frame: pd.DataFrame,
        members: Sequence[int],
        policy: str,
    ) -> int:
        """Index of the member that survives the group.

        ``members[0]`` is the group's representative, and representatives are
        chosen best-quality-first in :meth:`_quality_order`.
        """
        if policy == "first":
            return min(members, key=lambda position: (int(frame["frame_idx"].iloc[position]), position))
        if policy == "largest":
            return max(
                members,
                key=lambda position: int(frame["width"].iloc[position]) * int(frame["height"].iloc[position]),
            )
        return members[0]
