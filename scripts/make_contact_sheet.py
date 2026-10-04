#!/usr/bin/env python
"""Build a contact sheet: one row per class, N samples per row.

Handy for eyeballing an exported dataset (or any folder of images) without
opening a file browser::

    # an exported dataset: one row per class
    python scripts/make_contact_sheet.py -c configs/demo.yaml --per-class 6 \\
        --output docs/demo_samples.png

    # any folder of images (e.g. the crops of one video, one tile per box)
    python scripts/make_contact_sheet.py --from-dir data/yolo_person/crops/video_22b577ac \\
        --per-class 16 --output data/yolo_person/crops_overview.png

Tiles are taken as evenly spaced samples rather than the first N, so the sheet
shows the spread and not just the opening seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdb.config import load_config  # noqa: E402
from acdb.utils.image import imread  # noqa: E402

TILE = 160
PAD = 6
LABEL_HEIGHT = 20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--dataset", default=None, help="override paths.dataset_dir")
    parser.add_argument("--from-dir", default=None,
                        help="build the sheet from any folder of images instead of a dataset")
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--output", "-o", type=Path, default=REPO_ROOT / "docs" / "demo_samples.png")
    parser.add_argument("--tile", type=int, default=TILE)
    args = parser.parse_args(argv)

    if args.from_dir:
        directory = Path(args.from_dir)
        if not directory.is_dir():
            print(f"no such directory: {directory}")
            return 1
        images = sorted(
            path for path in directory.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not images:
            print(f"no images under {directory}")
            return 1
        step = max(1, len(images) // (args.per_class * 4))
        picked = images[::step]
        rows = [picked[index : index + args.per_class] for index in range(0, len(picked), args.per_class)]
        row_count = len(rows)
        tile = args.tile
        width = PAD + args.per_class * (tile + PAD)
        height = PAD + 18 + row_count * (tile + PAD) + PAD
        sheet = np.full((height, width, 3), 28, dtype=np.uint8)
        cv2.putText(sheet, f"{directory.name}: {len(images)} image(s), showing {len(picked)}",
                    (PAD, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        for index, path in enumerate(picked):
            row, column = divmod(index, args.per_class)
            x0 = PAD + column * (tile + PAD)
            y0 = PAD + 18 + row * (tile + PAD)
            sheet[y0 : y0 + tile, x0 : x0 + tile] = cv2.resize(
                imread(path), (tile, tile), interpolation=cv2.INTER_AREA
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), sheet)
        print(f"wrote {args.output} ({width}x{height}, {len(picked)} of {len(images)} images)")
        return 0

    cfg = load_config(args.config, root=args.root)
    dataset = Path(args.dataset).resolve() if args.dataset else cfg.paths.dataset_dir

    rows: list[tuple[str, list[tuple[np.ndarray, str]]]] = []
    for split in ("train", "val", "test"):
        split_dir = dataset / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            images = sorted(class_dir.glob("*.jpg"))
            if not images:
                continue
            step = max(1, len(images) // args.per_class)
            picked = images[::step][: args.per_class]
            tiles = [(imread(path), f"{split} {path.stem.split('_f')[-1]}") for path in picked]
            for existing, entries in rows:
                if existing == class_dir.name:
                    entries.extend(tiles)
                    break
            else:
                rows.append((class_dir.name, tiles))

    if not rows:
        print(f"no images found under {dataset}")
        return 1

    tile = args.tile
    width = PAD + args.per_class * (tile + PAD)
    height = len(rows) * (tile + LABEL_HEIGHT + PAD) + PAD
    sheet = np.full((height, width, 3), 28, dtype=np.uint8)

    for row_index, (name, tiles) in enumerate(rows):
        y0 = PAD + row_index * (tile + LABEL_HEIGHT + PAD)
        cv2.putText(sheet, name, (PAD, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (235, 235, 235), 1, cv2.LINE_AA)
        for column, (image, caption) in enumerate(tiles[: args.per_class]):
            x0 = PAD + column * (tile + PAD)
            y1 = y0 + LABEL_HEIGHT
            resized = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
            sheet[y1 : y1 + tile, x0 : x0 + tile] = resized
            cv2.putText(sheet, caption, (x0 + 2, y1 + tile - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (60, 240, 60), 1, cv2.LINE_AA)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(f"wrote {args.output} ({width}x{height}, {len(rows)} classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
