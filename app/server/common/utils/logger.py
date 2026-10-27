# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import logging
import logging.config
import os
from copy import deepcopy
from pathlib import Path
import sys

###############################################################################
class UnicodeSafeFormatter(logging.Formatter):

    # -------------------------------------------------------------------------
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        stream_encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
        return message.encode(stream_encoding, errors="backslashreplace").decode(
            stream_encoding,
            errors="strict",
        )


class AccessLogFilter(logging.Filter):
    """Keep routine successful requests at DEBUG and failures visible."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not isinstance(record.args, tuple) or len(record.args) != 5:
            return True
        _, method, target, _, status = record.args
        try:
            status = int(status)
        except (TypeError, ValueError):
            return True
        path = str(target).split("?", 1)[0]
        if status >= 500:
            record.levelno, record.levelname = logging.ERROR, "ERROR"
        elif status >= 400:
            record.levelno, record.levelname = logging.WARNING, "WARNING"
        elif status < 400 and (
            method in {"GET", "HEAD", "OPTIONS"}
            or (method == "POST" and path in {
                "/api/v1/datasets/import/preview",
                "/api/v1/datasets/import/validate",
            })
        ):
            record.levelno, record.levelname = logging.DEBUG, "DEBUG"
            return os.getenv("ADSMOD_LOG_LEVEL", "INFO").upper() == "DEBUG"
        return True


def uvicorn_logging_config() -> dict:
    """Extend Uvicorn's standard configuration without adding duplicate handlers."""
    from uvicorn.config import LOGGING_CONFIG

    config = deepcopy(LOGGING_CONFIG)
    config["filters"] = {"access_policy": {"()": AccessLogFilter}}
    config["handlers"]["access"]["filters"] = ["access_policy"]
    return config


logger = logging.getLogger("ADSMOD")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        logging.DEBUG if os.getenv("ADSMOD_LOG_LEVEL", "INFO").upper() == "DEBUG" else logging.INFO
    )
    console_handler.setFormatter(UnicodeSafeFormatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
logger.propagate = False

###############################################################################
def configure_logging(log_directory: Path | None = None) -> None:
    """Configure optional Core-owned file logging below the supplied storage root."""

    if log_directory is None:
        return
    try:
        log_directory.mkdir(parents=True, exist_ok=True)
        filename = (log_directory / "ADSMOD.log").resolve()
        for handler in list(logger.handlers):
            if not isinstance(handler, logging.FileHandler):
                continue
            if Path(handler.baseFilename).resolve() == filename:
                return
            logger.removeHandler(handler)
            handler.close()
        if not any(
            isinstance(handler, logging.FileHandler) for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(filename, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)
    except OSError:
        logger.warning(
            "Core file logging is unavailable; continuing with console logging."
        )

###############################################################################
def close_file_logging() -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()


__all__ = ["UnicodeSafeFormatter", "close_file_logging", "configure_logging", "logger"]
