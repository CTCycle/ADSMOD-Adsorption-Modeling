"""Pytest configuration and shared fixtures for ADSMOD E2E tests."""

from __future__ import annotations

import os

import pytest
from playwright.sync_api import APIRequestContext, Page, Playwright


# [CONSTANTS]
###############################################################################
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(TESTS_DIR, "fixtures")
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SETTINGS_ENV = os.path.join(PROJECT_ROOT, "ADSMOD", "settings", ".env")


# -------------------------------------------------------------------------
def load_env_values(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    if not os.path.exists(path):
        return values

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            key, separator, value = line.partition("=")
            if not separator:
                continue
            cleaned_key = key.strip()
            cleaned_value = value.strip()
            if (
                len(cleaned_value) >= 2
                and cleaned_value[0] == cleaned_value[-1]
                and cleaned_value[0] in {'"', "'"}
            ):
                cleaned_value = cleaned_value[1:-1]
            values[cleaned_key] = cleaned_value
    return values


# -------------------------------------------------------------------------
def resolve_test_urls() -> tuple[str, str]:
    env_values = load_env_values(SETTINGS_ENV)
    frontend_host = env_values.get("UI_HOST", "127.0.0.1")
    frontend_port = env_values.get("UI_PORT", "7861")
    backend_host = env_values.get("FASTAPI_HOST", "127.0.0.1")
    backend_port = env_values.get("FASTAPI_PORT", "8000")

    frontend_url = os.getenv(
        "ADSMOD_TEST_FRONTEND_URL", f"http://{frontend_host}:{frontend_port}"
    )
    backend_url = os.getenv(
        "ADSMOD_TEST_BACKEND_URL", f"http://{backend_host}:{backend_port}"
    )
    return frontend_url.rstrip("/"), backend_url.rstrip("/")


FRONTEND_URL, BACKEND_URL = resolve_test_urls()


###############################################################################
@pytest.fixture(scope="session")
def base_url() -> str:
    """Return the frontend base URL."""
    return FRONTEND_URL


# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Return the backend API base URL."""
    return BACKEND_URL


# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def api_context(playwright: Playwright, api_base_url: str) -> APIRequestContext:
    """Create a Playwright API request context for backend calls."""
    context = playwright.request.new_context(base_url=api_base_url)
    yield context
    context.dispose()


# -----------------------------------------------------------------------------
@pytest.fixture(scope="function")
def page(playwright: Playwright, base_url: str) -> Page:
    """Create a new browser page for each test."""
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()
    browser.close()


# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_csv_path() -> str:
    """Return the path to the sample adsorption CSV fixture."""
    return os.path.join(FIXTURES_DIR, "sample_adsorption.csv")


# -----------------------------------------------------------------------------
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Ensure heavy training/data tests run last."""
    heavy_items: list[pytest.Item] = []
    regular_items: list[pytest.Item] = []

    for item in items:
        nodeid = item.nodeid.lower()
        if "backend/performance" in nodeid or "training_perf" in nodeid:
            heavy_items.append(item)
        else:
            regular_items.append(item)

    items[:] = regular_items + heavy_items
