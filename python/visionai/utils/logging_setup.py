"""Logging configuration and small shared utilities."""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from visionai.config.paths import default_log_dir

LOGGER_NAME = "visionai"
_throttle_lock = threading.Lock()
_last_logged: dict[str, float] = defaultdict(float)

_console_formatter = logging.Formatter(
    "%(asctime)s  %(levelname)-7s %(name)-28s %(message)s", datefmt="%H:%M:%S"
)
_file_formatter = logging.Formatter(
    "%(asctime)s  %(levelname)-7s %(name)-28s %(filename)s:%(lineno)d  %(message)s"
)


def log_throttled(key: str, message: str, interval: float = 10.0, level: int = logging.WARNING) -> None:
    """Log ``message`` at most once per ``interval`` seconds for a given ``key``.

    Used for conditions that repeat every frame (a sensor that is missing, a
    device that is busy) where a log line per frame would drown everything else.
    """
    now = time.monotonic()
    with _throttle_lock:
        if now - _last_logged[key] < interval:
            return
        _last_logged[key] = now
    logging.getLogger(LOGGER_NAME).log(level, message)


def configure_logging(
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
    directory: str | Path | None = None,
    max_bytes: int = 2_000_000,
    backups: int = 3,
) -> logging.Logger:
    """Set up console and rotating-file handlers exactly once."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(_console_formatter)
        logger.addHandler(stream)

    if file:
        try:
            target = Path(directory) if directory else default_log_dir()
            target.mkdir(parents=True, exist_ok=True)
            rotating = logging.handlers.RotatingFileHandler(
                target / "visionai.log",
                maxBytes=max_bytes,
                backupCount=backups,
                encoding="utf-8",
            )
            rotating.setFormatter(_file_formatter)
            logger.addHandler(rotating)
        except OSError as exc:
            logger.warning("file logging disabled: %s", exc)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    logging.getLogger("PIL").setLevel(logging.WARNING)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the application root."""
    short = name.split(".")[-1] if name.startswith("visionai.") else name
    return logging.getLogger(f"{LOGGER_NAME}.{short}")


def describe_exception(exc: BaseException) -> dict[str, Any]:
    """Structured representation of an exception for logs and the error panel."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "module": type(exc).__module__,
    }
