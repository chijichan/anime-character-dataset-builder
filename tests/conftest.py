"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIGS = REPO_ROOT / "configs"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return CONFIGS


@pytest.fixture()
def demo_config(tmp_path: Path):
    """The demo configuration re-rooted into a temporary directory."""
    from acdb.config import load_config

    return load_config(
        CONFIGS / "demo.yaml",
        root=tmp_path,
        overrides=[
            "paths.input_dir=input",
            "paths.work_dir=work",
            "paths.frames_dir=work/frames",
            "paths.crops_dir=work/crops",
            "paths.rejected_dir=work/rejected",
            "paths.dataset_dir=work/dataset",
            "paths.metadata_dir=work/metadata",
            "paths.clusters_dir=work/clusters",
            "paths.logs_dir=work/logs",
            "export.stats_path=work/metadata/class_stats.csv",
        ],
    )
