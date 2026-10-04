"""Exception hierarchy shared by every ACDB module.

Keeping the exceptions in one place lets the CLI translate them into clean
one-line error messages (instead of tracebacks) while still allowing tests to
assert on specific failure modes.
"""

from __future__ import annotations

__all__ = [
    "AcbdError",
    "ConfigError",
    "StageError",
    "StateError",
    "MetadataError",
    "ModelUnavailableError",
]


class AcbdError(Exception):
    """Base class for all errors raised by ACDB."""


class ConfigError(AcbdError):
    """The YAML configuration is missing, malformed or has unknown keys."""


class StageError(AcbdError):
    """A pipeline stage failed; the message names the stage and the cause."""


class StateError(AcbdError):
    """The ``.state/`` bookkeeping is missing, corrupt or out of sync."""


class MetadataError(AcbdError):
    """A metadata table (CSV) or label file violates the output contract."""


class ModelUnavailableError(AcbdError):
    """An optional backend (ultralytics / timm / onnxruntime) is not installed.

    Raised with an actionable message telling the user which extra to install
    or which fallback ``type`` to configure.
    """
