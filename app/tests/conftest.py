# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Pytest configuration and shared fixtures for ADSMOD E2E tests."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from playwright.sync_api import APIRequestContext, Page, Playwright
from server.configurations.bootstrap import (
    load_database_config,
    load_storage_root,
    resolve_settings_database_path,
)
from server.configurations.settings import RuntimeConfig

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
APP_ROOT = TESTS_DIR.parent
REPOSITORY_ROOT = APP_ROOT.parent
WILDCARD_BIND_HOSTS = {"", "0.0.0.0", "::", "[::]"}

###############################################################################
def normalize_client_host(bind_host: str) -> str:
    stripped = bind_host.strip()
    return "127.0.0.1" if stripped in WILDCARD_BIND_HOSTS else stripped

###############################################################################
def resolve_test_urls() -> tuple[str, str]:
    database = load_database_config()
    storage_root = load_storage_root()
    settings_path = resolve_settings_database_path(database, storage_root)
    runtime = RuntimeConfig()
    if settings_path.is_file():
        try:
            connection = sqlite3.connect(f"file:{settings_path.as_posix()}?mode=ro", uri=True)
            try:
                row = connection.execute(
                    "SELECT runtime_host, runtime_backend_port, runtime_frontend_port "
                    "FROM application_settings WHERE id = 1"
                ).fetchone()
            finally:
                connection.close()
            if row is not None:
                runtime = RuntimeConfig(
                    host=row[0], backend_port=row[1], frontend_port=row[2]
                )
        except sqlite3.Error:
            # Test collection must not initialize or mutate the user's settings DB.
            pass
    host = normalize_client_host(runtime.host)
    return (
        f"http://{host}:{runtime.frontend_port}".rstrip("/"),
        f"http://{host}:{runtime.backend_port}".rstrip("/"),
    )


FRONTEND_URL, BACKEND_URL = resolve_test_urls()

###############################################################################
@pytest.fixture(scope="session")
def base_url() -> str:
    return FRONTEND_URL

###############################################################################
@pytest.fixture(scope="session")
def api_base_url() -> str:
    return BACKEND_URL

###############################################################################
@pytest.fixture(scope="session")
def api_context(playwright: Playwright, api_base_url: str) -> APIRequestContext:
    context = playwright.request.new_context(base_url=api_base_url)
    yield context
    context.dispose()

###############################################################################
@pytest.fixture(scope="session")
def ml_api_base_url(api_base_url: str) -> str:
    """Compatibility fixture for ML-focused tests, using the one canonical backend."""
    return api_base_url

###############################################################################
@pytest.fixture(scope="session")
def ml_api_context(playwright: Playwright, api_base_url: str) -> APIRequestContext:
    context = playwright.request.new_context(base_url=api_base_url)
    capabilities = context.get("/api/v1/system/capabilities")
    if not capabilities.ok or not capabilities.json().get("features", {}).get("machine_learning"):
        context.dispose()
        pytest.skip("Optional machine learning dependencies are not installed.")
    yield context
    context.dispose()

###############################################################################
@pytest.fixture
def page_context(page: Page, base_url: str) -> Page:
    page.goto(base_url)
    return page

###############################################################################
@pytest.fixture(scope="session")
def sample_csv_path() -> Path:
    return FIXTURES_DIR / "sample_adsorption.csv"
