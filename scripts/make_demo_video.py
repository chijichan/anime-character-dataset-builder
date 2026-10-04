#!/usr/bin/env python
"""Render the synthetic demo footage used by ``configs/demo.yaml``.

Thin wrapper around :func:`acdb.demo.make_demo_videos` so the generator can be
run without installing the console script::

    python scripts/make_demo_video.py --output data/input

It writes three short "episodes" (one per synthetic show), a
``reference/<anime>__<character>__<view>.jpg`` sheet per character, and
``demo_ground_truth.json`` describing both. Nothing is downloaded and no
copyrighted material is involved - the frames are drawn procedurally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdb.demo import DEMO_CHARACTERS, make_demo_videos  # noqa: E402
from acdb.utils.logging import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", "-o", type=Path, default=REPO_ROOT / "data" / "input",
                        help="directory that will hold the generated videos")
    parser.add_argument("--seconds", type=float, default=12.0, help="length of each episode")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--seed", type=int, default=20240517)
    args = parser.parse_args(argv)

    setup_logging(logs_dir=REPO_ROOT / "logs", level="INFO")
    manifest = make_demo_videos(
        args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds=args.seconds,
        seed=args.seed,
    )
    print(f"\nwrote {len(manifest['videos'])} video(s) to {args.output}")
    for name in sorted(manifest["videos"]):
        characters = ", ".join(item["character"] for item in manifest["videos"][name])
        print(f"  {name}: {characters}")
    print(f"characters: {len(DEMO_CHARACTERS)} across 3 shows")
    print(f"ground truth: {args.output / 'demo_ground_truth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
