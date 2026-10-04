"""Anime Character Dataset Builder (ACDB).

A modular pipeline that turns raw anime video into an ImageFolder-style
character classification dataset:

    extract-frames -> detect -> crop -> dedup -> filter -> cluster
                   -> apply-labels -> split -> export

The package is organised as:

``acdb.config``       typed YAML configuration loading/validation
``acdb.utils``        naming, logging, state, device, video and image helpers
``acdb.io``           metadata CSV / classes.json / cluster label persistence
``acdb.detectors``    pluggable :class:`~acdb.detectors.base.BaseDetector`
``acdb.extractors``   pluggable :class:`~acdb.extractors.base.BaseFeatureExtractor`
``acdb.clustering``   pluggable :class:`~acdb.clustering.base.BaseClusterer`
``acdb.pipeline``     the nine pipeline stages plus the resume-aware runner
``acdb.cli``          the ``acdb`` Typer command line interface

Only research/study use is intended; see the README for the compliance notice.
"""

from __future__ import annotations

__all__ = ["__version__", "SCHEMA_VERSION"]

__version__ = "0.1.0"

#: Bumped whenever on-disk artefacts (metadata columns, state files) change in
#: an incompatible way; stored in every ``.state/{stage}.json`` so that an old
#: cache is invalidated instead of silently mis-read.
SCHEMA_VERSION = 1
