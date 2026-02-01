"""Logging configuration for paperflow."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(".logs")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_RETENTION_DAYS = 7


def _get_log_file(log_dir: Path) -> Path:
    """Get the log file path for today, handling permission issues.

    Uses date-stamped filenames: paperflow.2024-01-31.log
    If the file exists but isn't writable (e.g., created by Docker as root),
    falls back to a unique filename.

    Args:
        log_dir: Directory for log files.

    Returns:
        Path to the log file.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"paperflow.{today}.log"

    # Check if file exists and is writable
    if log_file.exists():
        if os.access(log_file, os.W_OK):
            return log_file
        # File exists but not writable - use fallback with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return log_dir / f"paperflow.{timestamp}.log"

    # File doesn't exist - check if we can create it
    try:
        log_file.touch()
        return log_file
    except PermissionError:
        # Can't create in this directory - use fallback
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return log_dir / f"paperflow.{timestamp}.log"


def _cleanup_old_logs(log_dir: Path) -> None:
    """Remove log files older than LOG_RETENTION_DAYS.

    Args:
        log_dir: Directory containing log files.
    """
    if not log_dir.exists():
        return

    cutoff = datetime.now().timestamp() - (LOG_RETENTION_DAYS * 86400)

    for log_file in log_dir.glob("paperflow.*.log"):
        try:
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()
        except (OSError, PermissionError):
            pass  # Skip files we can't access


def setup_logging(
    level: int = logging.INFO,
    log_dir: Path | None = None,
    verbose: bool = False,
) -> None:
    """Configure logging for paperflow.

    Sets up:
    - File logging with date-stamped filenames (paperflow.YYYY-MM-DD.log)
    - Automatic cleanup of logs older than 7 days
    - Graceful handling of permission issues (e.g., root-owned files from Docker)
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

    # Clean up old log files
    _cleanup_old_logs(log_dir)

    # File handler with date-stamped filename
    log_file = _get_log_file(log_dir)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
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
