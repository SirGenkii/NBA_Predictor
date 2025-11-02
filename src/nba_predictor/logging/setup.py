from __future__ import annotations

import logging
from logging.config import dictConfig
from pathlib import Path
from typing import Any, Dict, Mapping

import structlog

DEFAULT_PROCESSORS = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.JSONRenderer(),
]


def _build_logging_config(
    log_path: Path,
    level: int,
    logger_name: str,
) -> Dict[str, Any]:
    """Return a dictionary config for the standard logging module."""

    log_path.parent.mkdir(parents=True, exist_ok=True)

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {"format": "%(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "plain",
                "level": level,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "plain",
                "filename": str(log_path),
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "level": level,
            },
        },
        "loggers": {
            logger_name: {
                "handlers": ["console", "file"],
                "level": level,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": level,
        },
    }


def configure_logging(
    log_dir: Path | str,
    *,
    pipeline: str | None = None,
    level: int = logging.INFO,
) -> None:
    """
    Configure both stdlib logging and structlog.

    Parameters
    ----------
    log_dir:
        Directory where log files should be stored.
    pipeline:
        Optional pipeline name used to derive the log filename.
    level:
        Logging level applied to handlers and loggers.
    """

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    log_filename = f"{pipeline}.log" if pipeline else "nba_predictor.log"
    log_path = log_dir_path / log_filename

    dictConfig(_build_logging_config(log_path, level, logger_name="nba_predictor"))

    structlog.configure(
        processors=DEFAULT_PROCESSORS,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

