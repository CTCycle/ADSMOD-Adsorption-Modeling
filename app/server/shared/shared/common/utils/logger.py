from __future__ import annotations

import logging
import logging.config
import sys
from datetime import datetime

from shared.common.paths import LOGS_DIR


###############################################################################
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = LOGS_DIR / f"ADSMOD_{current_timestamp}.log"

try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_filename.touch(exist_ok=True)
    file_log_available = True
except OSError:
    file_log_available = False

###############################################################################
class UnicodeSafeFormatter(logging.Formatter):

    # -------------------------------------------------------------------------
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        stream_encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
        return message.encode(stream_encoding, errors="backslashreplace").decode(
            stream_encoding, errors="strict"
        )

###############################################################################
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
    },
    "loggers": {
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"],
    },
}

if file_log_available:
    LOG_CONFIG["handlers"]["file"] = {
        "class": "logging.FileHandler",
        "level": "DEBUG",
        "formatter": "default",
        "filename": str(log_filename),
        "mode": "a",
        "encoding": "utf-8",
    }
    LOG_CONFIG["root"]["handlers"].append("file")

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("ADSMOD")
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(UnicodeSafeFormatter("%(levelname)s - %(message)s"))

