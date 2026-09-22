"""
logger.py – Centralized Logging for the Adaptive RAG Project
==============================================================
Provides a pre-configured logger that writes to both the console and
a log file.  Every module imports ``logger`` from here so that all
messages appear in one place.

Usage
-----
    from utils.logger import logger

    logger.info("Pipeline started")
    logger.error("Something went wrong", exc_info=True)

Author  : B.Tech CSE-AI Student Project
"""

import io
import logging
import sys
from typing import Optional

from utils.config import LOG_FILE


def _utf8_stdout() -> io.TextIOWrapper:
    """Return a UTF-8 encoded wrapper around stdout for safe Unicode logging on Windows."""
    return io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )


def setup_logger(
    name: str = "AdaptiveRAG",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger with console + file handlers.

    Parameters
    ----------
    name : str
        Name of the logger (appears in every log line).
    log_file : str | None
        Path to the log file.  Defaults to the path in config.py.
    level : int
        Minimum severity level to capture (default: INFO).

    Returns
    -------
    logging.Logger
        Fully configured logger instance.
    """
    # Use the path from config if none is given
    if log_file is None:
        log_file = str(LOG_FILE)

    # Grab or create the named logger
    created_logger = logging.getLogger(name)

    # Avoid duplicate handlers when this function is called more than once
    if created_logger.handlers:
        return created_logger

    created_logger.setLevel(level)

    # ── Formatter ────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s │ %(name)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (stdout, UTF-8 safe) ─────────────────────────
    console_handler = logging.StreamHandler(_utf8_stdout())
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    created_logger.addHandler(console_handler)

    # ── File handler ─────────────────────────────────────────────────
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        created_logger.addHandler(file_handler)
    except (OSError, PermissionError) as exc:
        created_logger.warning("Could not create log file %s: %s", log_file, exc)

    return created_logger


# ── Module-level logger (import this everywhere) ─────────────────────
logger: logging.Logger = setup_logger()
