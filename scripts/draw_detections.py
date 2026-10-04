#!/usr/bin/env python
"""Draw recorded detection boxes onto their source frames, for eyeballing.

``detect`` writes boxes into ``detections.csv``; this renders them back onto the
frames so you can check recall and false positives without writing any code::

    python scripts/draw_detections.py -c configs/yolo_person.yaml
    python scripts/draw_detections.py -c configs/yolo_person.yaml --only-multi --limit 30

Output goes to ``<paths.work_dir>/annotated/`` (next to frames/ and crops/), one
JPEG per frame that has at least one box. Green boxes are at or above
``detect.filter.min_conf``, amber ones below the *model* threshold of 0.35, so a
glance tells you whether the low-confidence detections are worth keeping.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdb.config import load_config  # noqa: E402
from acdb.io import read_table  # noqa: E402
from acdb.utils.image import imread  # noqa: E402

GREEN = (0, 255, 0)
AMBER = (0, 200, 255)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--limit", type=int, default=0, help="render at most N frames (0 = all)")
    parser.add_argument("--only-multi", action="store_true", help="only frames with 2+ boxes")
    parser.add_argument("--highlight", type=float, default=0.35,
                        help="confidence below which a box is drawn in amber")
    parser.add_argument("--width", type=int, default=960, help="output width in pixels")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, root=args.root)
    table_path = cfg.paths.metadata_dir / "detections.csv"
    if not table_path.is_file():
        print(f"no detections yet: {table_path} (run `acdb detect` first)")
        return 1

    detections = read_table(table_path, table="detections")
    out_dir = cfg.paths.work_dir / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_frame = detections.groupby("frame_id").size().sort_values(ascending=False)
    if args.only_multi:
        per_frame = per_frame[per_frame >= 2]
    if args.limit:
        per_frame = per_frame.head(args.limit)

    by_frame = {frame_id: group for frame_id, group in detections.groupby("frame_id")}
    written = 0
    for frame_id in per_frame.index:
        group = by_frame[frame_id]
        source = cfg.root / str(group.iloc[0]["frame_path"])
        image = imread(source)
        height, width = image.shape[:2]
        scale = min(1.0, args.width / float(width))
        if scale < 1.0:
            image = cv2.resize(image, (int(width * scale), int(height * scale)),
                               interpolation=cv2.INTER_AREA)
        for record in group.to_dict("records"):
            x1 = int(int(record["bbox_x1"]) * scale)
            y1 = int(int(record["bbox_y1"]) * scale)
            x2 = int(int(record["bbox_x2"]) * scale)
            y2 = int(int(record["bbox_y2"]) * scale)
            confidence = float(record["det_conf"])
            colour = GREEN if confidence >= args.highlight else AMBER
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(image, f"{confidence:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        caption = (f"{frame_id}  boxes={len(group)}  "
                   f"t={int(group.iloc[0]['timestamp_ms']) / 1000:.1f}s")
        cv2.putText(image, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{frame_id}.jpg"), image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        written += 1

    print(f"wrote {written} annotated frame(s) to {out_dir}")
    print(
        f"  green = conf >= {args.highlight:.2f}, "
        f"amber = lower (kept by detect.filter.min_conf={cfg.detect.filter.min_conf:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
