"""Pytest configuration and shared fixtures for ADSMOD E2E tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from playwright.sync_api import APIRequestContext, Page, Playwright


# [CONSTANTS]
###############################################################################
TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
APP_ROOT = TESTS_DIR.parent
WILDCARD_BIND_HOSTS = {"", "0.0.0.0", "::", "[::]"}
CANONICAL_CONFIG = APP_ROOT / "resources" / "adsmod.json"

###############################################################################
def normalize_client_host(bind_host: str) -> str:
    """Convert wildcard bind hosts into a routable client host."""
    stripped = bind_host.strip()
    if stripped in WILDCARD_BIND_HOSTS:
        return "127.0.0.1"
    return stripped

###############################################################################
def resolve_test_urls() -> tuple[str, str, str]:
    with CANONICAL_CONFIG.open("r", encoding="utf-8") as handle:
        runtime = json.load(handle)["runtime"]
    frontend_host = normalize_client_host(runtime["host"])
    frontend_port = str(runtime["frontend_port"])
    backend_host = normalize_client_host(runtime["host"])
    backend_port = str(runtime["core_port"])
    ml_host = normalize_client_host(runtime["host"])
    ml_port = str(runtime["ml_port"])

    frontend_url = f"http://{frontend_host}:{frontend_port}"
    backend_url = f"http://{backend_host}:{backend_port}"
    ml_backend_url = f"http://{ml_host}:{ml_port}"
    return frontend_url.rstrip("/"), backend_url.rstrip("/"), ml_backend_url.rstrip("/")


FRONTEND_URL, BACKEND_URL, ML_BACKEND_URL = resolve_test_urls()

###############################################################################
@pytest.fixture(scope="session")
def base_url() -> str:
    """Return the frontend base URL."""
    return FRONTEND_URL

###############################################################################
@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Return the backend API base URL."""
    return BACKEND_URL

###############################################################################
@pytest.fixture(scope="session")
def api_context(playwright: Playwright, api_base_url: str) -> APIRequestContext:
    """Create a Playwright API request context for backend calls."""
    context = playwright.request.new_context(base_url=api_base_url)
    yield context
    context.dispose()

###############################################################################
@pytest.fixture(scope="session")
def ml_api_base_url() -> str:
    """Return the ML backend API base URL."""
    return ML_BACKEND_URL

###############################################################################
@pytest.fixture(scope="session")
def ml_api_context(playwright: Playwright, ml_api_base_url: str) -> APIRequestContext:
    """Create a Playwright API request context for ML backend calls."""
    with CANONICAL_CONFIG.open("r", encoding="utf-8") as handle:
        runtime_mode = json.load(handle)["runtime"]["mode"]
    if runtime_mode != "core-ml":
        pytest.skip("ML E2E tests require runtime.mode=core-ml.")
    context = playwright.request.new_context(base_url=ml_api_base_url)
    yield context
    context.dispose()

###############################################################################
@pytest.fixture(scope="function")
def page(playwright: Playwright, base_url: str) -> Page:
    """Create a new browser page for each test."""
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()
    browser.close()

###############################################################################
@pytest.fixture(scope="session")
def sample_csv_path() -> str:
    """Return the path to the sample adsorption CSV fixture."""
    return str(FIXTURES_DIR / "sample_adsorption.csv")

###############################################################################
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
