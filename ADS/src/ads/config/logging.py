"""Logging configuration for ADS.

Provides structured JSON logging for production and
rich console logging for development.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

from ads.config.settings import settings


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "props"):
            log_entry["props"] = record.props

        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging() -> None:
    """Configure logging based on environment settings."""
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.value)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(settings.log_level.value)

    if settings.log_format == "json" and settings.environment == "production":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)

    logging.getLogger("ads").setLevel(settings.log_level.value)
