import logging

import pytest

from server.common.utils.logger import uvicorn_logging_config


@pytest.mark.parametrize(
    ("method", "path", "status", "debug", "expected"),
    [
        ("GET", "/health?probe=1", 200, False, None),
        ("GET", "/api/v1/fitting/jobs/123", 200, True, logging.DEBUG),
        ("POST", "/api/v1/datasets/import/preview", 200, False, None),
        ("POST", "/api/v1/datasets/import/validate", 200, True, logging.DEBUG),
        ("POST", "/api/v1/datasets/import/commit", 200, False, logging.INFO),
        ("GET", "/health", 404, False, logging.WARNING),
        ("POST", "/api/v1/datasets/import/preview", 500, False, logging.ERROR),
    ],
)
def test_access_log_policy(monkeypatch, method, path, status, debug, expected):
    monkeypatch.setenv("ADSMOD_LOG_LEVEL", "DEBUG" if debug else "INFO")
    config = uvicorn_logging_config()
    handler = logging.StreamHandler()
    handler.addFilter(config["filters"]["access_policy"]["()"]())
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %s',
        ("127.0.0.1", method, path, "1.1", status), None,
    )
    assert bool(handler.filter(record)) is (expected is not None)
    if expected is not None:
        assert record.levelno == expected

