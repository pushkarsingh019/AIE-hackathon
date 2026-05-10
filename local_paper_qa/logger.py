"""Logging configuration for LocalPaperQA.

Configures structured logging with levels for:
- DEBUG: detailed information for debugging
- INFO: general operational messages
- WARNING: warning messages for non-critical issues
- ERROR: error messages for failures

Usage::
    from local_paper_qa.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Indexed %d papers", count)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_DIR = Path("papers") / ".research_index" / "logs"


def get_logger(name: str = "local_paper_qa", level: str = "INFO") -> logging.Logger:
    """Get a configured logger for the given module name."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)
    
    # File handler (if logs directory exists/writable)
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(_LOG_DIR / "paper_qa.log")
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s %(filename)s:%(lineno)d: %(message)s"
        ))
        logger.addHandler(file_handler)
    except Exception:
        pass  # File logging is best-effort
    
    return logger
