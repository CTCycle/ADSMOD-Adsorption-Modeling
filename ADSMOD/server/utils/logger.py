from __future__ import annotations

import logging
import logging.config
import sys
from datetime import datetime
from os import makedirs
from os.path import join

from ADSMOD.server.utils.constants import LOGS_PATH


###############################################################################
makedirs(LOGS_PATH, exist_ok=True)
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = join(LOGS_PATH, f"ADSMOD_{current_timestamp}.log")


###############################################################################
class UnicodeSafeFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        stream_encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
        return message.encode(stream_encoding, errors="backslashreplace").decode(
            stream_encoding, errors="strict"
        )


LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%d-%m-%Y %H:%M:%S",
        },
        "minimal": {
            "format": "%(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "minimal",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "default",
            "filename": log_filename,
            "mode": "a",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
}

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("ADSMOD")
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(UnicodeSafeFormatter("%(levelname)s - %(message)s"))
