"""loguru wiring: coloured console sink + rotating file sink.

Every stage ends with :func:`stage_summary`, which prints the four numbers the
spec asks for - processed / filtered / produced / elapsed - in one line, so a
long run can be skimmed by searching for ``summary``.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional

from loguru import logger as _logger

__all__ = ["setup_logging", "get_logger", "stage_summary"]

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{extra[stage]}</cyan> | <level>{message}</level>"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[stage]} | "
    "{name}:{function}:{line} | {message}"
)

_CONFIGURED = False


def get_logger(stage: str = "acdb"):
    """Return a logger bound to a ``stage`` (used by the console/file format)."""
    return _logger.bind(stage=stage)


def setup_logging(
    *,
    logs_dir: Path,
    level: str = "INFO",
    file_level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: int = 14,
    colorize: bool = True,
    run_name: str = "pipeline",
) -> Any:
    """Install the console and file sinks (idempotent, replaces old sinks)."""
    global _CONFIGURED
    logs_dir.mkdir(parents=True, exist_ok=True)
    _logger.remove()
    _logger.configure(extra={"stage": "acdb"})
    _logger.add(
        sys.stderr,
        level=level.upper(),
        format=_CONSOLE_FORMAT,
        colorize=colorize,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )
    log_file = logs_dir / f"{run_name}_{date.today().isoformat()}.log"
    _logger.add(
        str(log_file),
        level=file_level.upper(),
        format=_FILE_FORMAT,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=False,
        backtrace=True,
        diagnose=False,
    )
    _CONFIGURED = True
    return _logger


def stage_summary(
    stage: str,
    *,
    processed: int,
    filtered: int,
    produced: int,
    seconds: float,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Log and return the standard per-stage summary line."""
    tail = ""
    if extra:
        tail = " " + " ".join(f"{k}={v}" for k, v in extra.items())
    line = (
        f"[{stage}] summary | processed={processed} filtered={filtered} "
        f"produced={produced} elapsed={seconds:.2f}s{tail}"
    )
    get_logger(stage).info(line)
    return line
