"""Stage 7 - ``apply-labels``: fold the human's cluster labels back in.

Reads ``data/clusters/cluster_labels.yaml`` and the review folder names (see
:mod:`acdb.io.labels`), maps every ``cluster_id`` onto
``{anime}__{character}``, and writes ``labeled.csv``.

Unlabelled clusters - and HDBSCAN noise (``cluster_id == -1``) - become
``unknown/unknown``: the spec's "未标注簇自动归入 unknown/" rule. They are
dropped from the final dataset only when ``export.drop_unknown`` is true, so a
review can always see what was left unlabelled.

The stage's config hash includes a hash of the label files, which means editing
``cluster_labels.yaml`` (or renaming a review folder) is enough to invalidate
this stage and everything downstream - no ``--force`` needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from acdb.config import AppConfig, hash_mapping
from acdb.errors import StageError
from acdb.io import UNKNOWN_KEY, read_labels
from acdb.pipeline.context import StageContext, StageOutcome
from acdb.pipeline.stage import Stage

__all__ = ["ApplyLabelsStage", "labels_revision"]

NOISE_CLUSTER = -1


def labels_revision(clusters_dir: Path) -> str:
    """Hash of everything a human can edit to define the labels."""
    directory = Path(clusters_dir)
    payload: Dict[str, object] = {}
    yaml_path = directory / "cluster_labels.yaml"
    if yaml_path.is_file():
        payload["yaml"] = yaml_path.read_text(encoding="utf-8")
    if directory.is_dir():
        payload["folders"] = sorted(entry.name for entry in directory.iterdir() if entry.is_dir())
    return hash_mapping(payload)


class ApplyLabelsStage(Stage):
    name = "apply_labels"
    section = "apply_labels"
    description = "merge human cluster labels into the metadata"

    @classmethod
    def config_payload(cls, cfg: AppConfig) -> Dict[str, object]:
        payload = dict(super().config_payload(cfg))
        payload["labels_revision"] = labels_revision(cfg.paths.clusters_dir)
        return payload

    def execute(self, ctx: StageContext) -> StageOutcome:
        clusters = ctx.load("clusters", required=["image_id", "cluster_id"])
        if clusters.empty:
            raise StageError("clusters.csv is empty - run `acdb cluster` first")

        labels = read_labels(ctx.cfg.paths.clusters_dir)
        if not labels:
            ctx.warning(
                f"no cluster label found in {ctx.rel(ctx.cfg.paths.clusters_dir)}; "
                f"every image will be unknown/unknown"
            )

        frame = clusters.copy()
        anime_column: List[str] = []
        character_column: List[str] = []
        labelled = unknown_clusters = 0
        for cluster_id in frame["cluster_id"].astype(int):
            label = labels.get(int(cluster_id))
            if int(cluster_id) == NOISE_CLUSTER or label is None or label.is_unknown:
                anime_column.append(UNKNOWN_KEY)
                character_column.append(UNKNOWN_KEY)
                continue
            anime_column.append(label.anime)
            character_column.append(label.character)
        frame["anime"] = anime_column
        frame["character"] = character_column

        present = {int(v) for v in frame["cluster_id"].unique()}
        for cluster_id in present:
            label = labels.get(cluster_id)
            if cluster_id != NOISE_CLUSTER and label is not None and not label.is_unknown:
                labelled += 1
            else:
                unknown_clusters += 1

        if not ctx.dry_run:
            ctx.save("labeled", frame)

        counts = (
            frame.assign(key=frame["anime"] + "__" + frame["character"])
            .groupby("key")
            .size()
            .sort_values(ascending=False)
        )
        ctx.info(
            f"labeled.csv: {len(frame)} image(s), {labelled} cluster(s) labelled, "
            f"{unknown_clusters} unknown"
        )
        for key, count in counts.items():
            ctx.info(f"  {key}: {count}")
        return StageOutcome(
            processed=len(frame),
            filtered=int((frame["anime"] == UNKNOWN_KEY).sum()),
            produced=labelled,
            items=frame["image_id"].astype(str).tolist(),
            outputs=[str(ctx.table("labeled"))],
            extra={"classes": int(counts.size), "unknown_images": int((frame["anime"] == UNKNOWN_KEY).sum())},
        )
