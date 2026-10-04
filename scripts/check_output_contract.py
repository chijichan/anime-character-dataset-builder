#!/usr/bin/env python
"""Validate a produced dataset against the hard output contract.

Checks, independently of the pipeline that produced it:

* ``data/metadata/metadata.csv`` has exactly the 16 contract columns, in order;
* every ``image_id`` is unique, no ``anime``/``character``/``split`` is empty,
  every ``split`` is one of ``train``/``val``/``test``;
* every ``file_path`` exists, sits in ``{split}/{anime}__{character}/`` and
  follows ``{anime}__{character}__{video_id}_f{frame:06d}_b{bbox:02d}.jpg``;
* every row's naming fields agree with its path and with ``video_id``;
* ``classes.json`` is well formed, ids are contiguous from 0, and every class
  directory on disk appears in it (and vice versa);
* reports the per-class sample counts as a table.

Exit code is 0 when everything holds, 1 otherwise, so it can gate CI::

    python scripts/check_output_contract.py --root . --config configs/demo.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdb.config import load_config  # noqa: E402
from acdb.io import (  # noqa: E402
    METADATA_COLUMNS,
    read_classes_json,
    read_table,
    validate_metadata,
)
from acdb.utils.naming import parse_final_filename  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--root", default=None, help="project root (default: cwd)")
    parser.add_argument("--dataset", default=None, help="override paths.dataset_dir")
    parser.add_argument("--metadata", default=None, help="override paths.metadata_dir")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, root=args.root)
    dataset_dir = Path(args.dataset).resolve() if args.dataset else cfg.paths.dataset_dir
    metadata_dir = Path(args.metadata).resolve() if args.metadata else cfg.paths.metadata_dir

    problems: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    metadata_path = metadata_dir / "metadata.csv"
    classes_path = metadata_dir / "classes.json"
    check(metadata_path.is_file(), f"missing {metadata_path}")
    check(classes_path.is_file(), f"missing {classes_path}")
    if problems:
        for problem in problems:
            print(f"FAIL  {problem}")
        return 1

    table = read_table(metadata_path, table="metadata")
    try:
        validate_metadata(table)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim
        problems.append(str(exc))

    check(list(table.columns) == list(METADATA_COLUMNS), "column order differs from the contract")
    check(len(table) > 0, "metadata.csv is empty")
    print(f"metadata.csv: {len(table)} row(s), {len(table.columns)} column(s)")

    for row in table.to_dict("records"):
        image = cfg.root / str(row["file_path"])
        name = image.name
        if not image.is_file():
            problems.append(f"missing file: {row['file_path']}")
            continue
        parsed = parse_final_filename(name)
        if parsed is None:
            problems.append(f"file name violates the contract: {name}")
            continue
        expected_dir = f"{row['anime']}__{row['character']}"
        if image.parent.name != expected_dir:
            problems.append(f"{name}: directory {image.parent.name} != class {expected_dir}")
        if image.parent.parent.name != row["split"]:
            problems.append(f"{name}: split directory does not match metadata")
        for field in ("anime", "character", "video_id"):
            if parsed[field] != row[field]:
                problems.append(f"{name}: {field} {parsed[field]!r} != metadata {row[field]!r}")
        if parsed["frame_idx"] != int(row["frame_idx"]):
            problems.append(f"{name}: frame_idx does not match metadata")

    classes = read_classes_json(classes_path)
    keys = sorted(entry.key for entry in classes)
    check([entry.id for entry in classes] == list(range(len(classes))), "class ids are not 0..n-1")
    check(len(set(keys)) == len(keys), "duplicate class in classes.json")
    print(f"classes.json: {len(classes)} class(es)")

    on_disk = {
        f"{split}/{path.parent.name}"
        for split in ("train", "val", "test")
        for path in (dataset_dir / split).glob("*/*.jpg")
    }
    on_disk_classes = sorted({entry.split("/", 1)[1] for entry in on_disk})
    check(on_disk_classes == keys, f"class directories {on_disk_classes} != classes.json {keys}")

    stats = Counter(
        f"{row['anime']}__{row['character']}" for row in table.to_dict("records")
    )
    splits = Counter(str(row["split"]) for row in table.to_dict("records"))
    print("\nsamples per class:")
    for key, count in stats.most_common():
        print(f"  {key:<32} {count:>5}")
    print("samples per split: " + ", ".join(f"{k}={v}" for k, v in sorted(splits.items())))

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems[:50]:
            print(f"  FAIL  {problem}")
        return 1
    print("\nOK: dataset satisfies the output contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
