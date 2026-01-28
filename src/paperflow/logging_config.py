"""Logging configuration for paperflow."""

import logging
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_DIR = Path(".logs")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_dir: Path | None = None,
    verbose: bool = False,
) -> None:
    """Configure logging for paperflow.

    Sets up:
    - File logging with 7-day rotation
    - Optional verbose console output (for --verbose flag)

    Args:
        level: Logging level (default: INFO).
        log_dir: Directory for log files (default: .logs).
        verbose: Whether to also log to console (for CLI --verbose mode).
    """
    log_dir = log_dir or LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create root logger for paperflow
    logger = logging.getLogger("paperflow")
    logger.setLevel(logging.DEBUG if verbose else level)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler with 7-day rotation
    log_file = log_dir / "paperflow.log"
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=7,  # Keep 7 days of logs
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(file_handler)

    # Console handler only in verbose mode
    if verbose:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(console_handler)

    logger.info(f"Logging initialized - log file: {log_file}, verbose: {verbose}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (will be prefixed with 'paperflow.').

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"paperflow.{name}")
