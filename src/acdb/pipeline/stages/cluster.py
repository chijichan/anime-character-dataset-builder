"""Stage 6 - ``cluster``: embed, cluster, and stage a human review.

Pipeline of this stage:

1. embed the surviving crops (cached in ``data/metadata/features.npz``);
2. optionally reduce the dimensionality (PCA by default - see
   :mod:`acdb.clustering.reduce` for why);
3. cluster (HDBSCAN by default; ``-1`` means *noise*);
4. export up to ``representatives_per_cluster`` images per cluster to
   ``data/clusters/cluster_{id:03d}/`` together with an ``index.csv``, and the
   noise points to ``data/clusters/noise/``;
5. write ``cluster_labels.yaml`` **only when it does not exist yet**, so a
   human's annotations always survive a re-run.

The human then either edits that YAML or renames a review folder (keeping the
``cluster_<id>`` token) and runs ``acdb apply-labels``.

Output: ``clusters.csv`` (``quality.csv`` survivors + ``cluster_id``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from acdb.clustering import build_clusterer, reduce_features
from acdb.errors import StageError
from acdb.extractors import build_extractor, compute_features
from acdb.io import cluster_folder_name, write_label_template
from acdb.pipeline.context import StageContext, StageOutcome
from acdb.pipeline.stage import Stage
from acdb.utils.device import describe_device, resolve_device
from acdb.utils.image import imread
from acdb.utils.logging import get_logger

__all__ = ["ClusterStage", "representative_indices", "suggest_anime"]


def representative_indices(vectors: np.ndarray, members: Sequence[int], limit: int) -> List[int]:
    """Indices of the ``limit`` members closest to the cluster centroid."""
    if not members:
        return []
    subset = np.asarray(vectors)[list(members)]
    centroid = subset.mean(axis=0)
    distances = np.linalg.norm(subset - centroid, axis=1)
    order = np.argsort(distances, kind="stable")
    return [int(members[position]) for position in order[: max(1, limit)]]


def suggest_anime(video_path: str, input_dir: Path, root: Path) -> str:
    """Guess the show name from the video's folder below ``input_dir``.

    ``data/input/demo_show_a/ep01.mp4`` -> ``demo_show_a``. Videos placed
    directly in the input directory yield an empty string (no suggestion).
    """
    candidate = Path(video_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.relative_to(input_dir)
    except ValueError:
        return ""
    parts = relative.parts[:-1]
    return parts[0] if parts else ""


class ClusterStage(Stage):
    name = "cluster"
    section = "cluster"
    extra_sections = ("features",)
    description = "embed crops, cluster them and export review folders"

    def execute(self, ctx: StageContext) -> StageOutcome:
        cfg = ctx.cfg
        settings = cfg.cluster
        logger = get_logger(self.name)

        quality = ctx.load("quality", required=["image_id", "crop_path", "kept"])
        candidates = quality.loc[quality["kept"].astype(bool)].copy()
        if candidates.empty:
            raise StageError("no crop survived the quality filter - nothing to cluster")

        device = resolve_device(cfg.project.device)
        extractor = build_extractor(cfg.features, device)
        ctx.info(f"extractor: {extractor.describe()} on {describe_device(device)}")
        lookup = dict(zip(candidates["image_id"].astype(str), candidates["crop_path"].astype(str)))
        feature_set = compute_features(
            list(lookup),
            loader=lambda image_id: imread(ctx.rel_abs(lookup[image_id])),
            extractor=extractor,
            cache_path=ctx.table("features"),
            batch_size=cfg.features.batch_size,
        )
        extractor.close()

        reduced, reduction = reduce_features(
            feature_set.vectors,
            settings.reduce_dim,
            settings.pca_components,
            seed=cfg.project.seed,
        )
        ctx.info(
            f"features: {feature_set.vectors.shape[0]}x{feature_set.vectors.shape[1]} -> "
            f"{reduced.shape[1]}D ({reduction.get('method')}"
            + (
                f", explained_variance={reduction.get('explained_variance', 0):.3f}"
                if reduction.get("explained_variance") is not None
                else ""
            )
            + ")"
        )

        clusterer = build_clusterer(settings, seed=cfg.project.seed)
        result = clusterer.fit_predict(reduced)
        effective = ", ".join(f"{k}={v}" for k, v in sorted((result.params or {}).items()))
        ctx.info(
            f"clusterer: {result.backend}({effective}) -> {result.n_clusters} cluster(s), "
            f"{result.n_noise} noise point(s)"
        )
        if result.n_clusters == 0:
            logger.warning(
                "no cluster was found; every image is noise. Try a smaller "
                "cluster.params.min_cluster_size or cluster.type=kmeans."
            )

        labels = result.labels.copy()
        if not settings.noise_as_unknown and result.n_clusters > 0:
            labels = self._absorb_noise(labels, reduced)

        frame = candidates.copy()
        frame["cluster_id"] = labels.astype(int)
        if not ctx.dry_run:
            ctx.save("clusters", frame)

        summary = self._export_review(ctx, frame, reduced, labels)
        if not ctx.dry_run:
            write_label_template(
                cfg.paths.clusters_dir,
                summary["clusters"],
                overwrite=False,
            )

        extra = {
            "clusters": result.n_clusters,
            "noise": int(np.count_nonzero(labels < 0)),
            "backend": result.backend,
        }
        silhouette = self._silhouette(reduced, labels)
        if silhouette is not None:
            extra["silhouette"] = round(silhouette, 4)
        ctx.info(
            f"review folders written to {cfg.paths.clusters_dir} - label them via "
            f"{ctx.rel(cfg.paths.clusters_dir / 'cluster_labels.yaml')} or by renaming "
            f"the folders, then run `acdb apply-labels`"
        )
        return StageOutcome(
            processed=len(frame),
            filtered=int(np.count_nonzero(labels < 0)),
            produced=result.n_clusters,
            items=frame["image_id"].astype(str).tolist(),
            outputs=[str(ctx.table("clusters")), str(cfg.paths.clusters_dir)],
            extra=extra,
        )

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _absorb_noise(labels: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Assign noise points to the nearest cluster centroid."""
        result = labels.copy()
        clusters = sorted({int(v) for v in labels if int(v) >= 0})
        if not clusters:
            return result
        centroids = np.stack([vectors[labels == c].mean(axis=0) for c in clusters])
        for position in np.nonzero(labels < 0)[0]:
            distances = np.linalg.norm(centroids - vectors[position], axis=1)
            result[position] = clusters[int(np.argmin(distances))]
        return result

    @staticmethod
    def _silhouette(vectors: np.ndarray, labels: np.ndarray) -> Optional[float]:
        clusters = {int(v) for v in labels if int(v) >= 0}
        if len(clusters) < 2 or vectors.shape[0] < 4 or vectors.shape[0] > 5000:
            return None
        try:  # pragma: no cover - depends on data
            from sklearn.metrics import silhouette_score

            mask = labels >= 0
            return float(silhouette_score(vectors[mask], labels[mask]))
        except Exception:  # pragma: no cover - defensive
            return None

    def _export_review(
        self,
        ctx: StageContext,
        frame: pd.DataFrame,
        reduced: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, object]:
        """Copy representative images into per-cluster review folders."""
        settings = ctx.cfg.cluster
        directory = ctx.cfg.paths.clusters_dir
        if ctx.dry_run:
            return {"clusters": {}}
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.iterdir():
            # Only refresh folders the pipeline owns; renamed ones are human work.
            if stale.is_dir() and (stale.name == "noise" or stale.name.startswith("cluster_")):
                shutil.rmtree(stale, ignore_errors=True)

        summary: Dict[int, Dict[str, object]] = {}
        records = frame.to_dict("records")
        for cluster_id in sorted({int(v) for v in labels}):
            positions = np.nonzero(labels == cluster_id)[0]
            chosen = representative_indices(
                reduced, positions.tolist(), settings.representatives_per_cluster
            )
            folder = directory / ("noise" if cluster_id < 0 else cluster_folder_name(cluster_id))
            folder.mkdir(parents=True, exist_ok=True)
            index_rows: List[dict] = []
            suggestions: Dict[str, int] = {}
            for position in chosen:
                record = records[position]
                source = ctx.rel_abs(str(record["crop_path"]))
                if not source.is_file():
                    continue
                target = folder / source.name
                if not target.exists():
                    shutil.copy2(source, target)
                anime = suggest_anime(
                    str(record.get("video_path", "")), ctx.cfg.paths.input_dir, ctx.cfg.root
                )
                if anime:
                    suggestions[anime] = suggestions.get(anime, 0) + 1
                index_rows.append(
                    {
                        "file": target.name,
                        "image_id": record["image_id"],
                        "video_path": record.get("video_path", ""),
                        "video_id": record.get("video_id", ""),
                        "frame_idx": record.get("frame_idx", ""),
                        "bbox_idx": record.get("bbox_idx", ""),
                        "det_conf": record.get("det_conf", ""),
                        "quality_score": record.get("quality_score", ""),
                        "suggested_anime": anime,
                    }
                )
            if index_rows:
                pd.DataFrame(index_rows).to_csv(
                    folder / "index.csv", index=False, encoding="utf-8", lineterminator="\n"
                )
            if cluster_id < 0:
                continue
            dominant = max(suggestions, key=suggestions.get) if suggestions else ""
            summary[cluster_id] = {
                "size": int(len(positions)),
                "suggested": "",
                "preview": next(
                    (str(row["video_path"]) for row in index_rows if row["video_path"]), ""
                ),
            }
            if dominant:
                summary[cluster_id]["suggested_hint"] = dominant
        return {"clusters": summary}
