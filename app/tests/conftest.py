# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Pytest configuration and shared fixtures for ADSMOD E2E tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import APIRequestContext, Page, Playwright
from server.configurations.bootstrap import load_runtime_settings

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"


###############################################################################
def resolve_backend_url() -> str:
    return load_runtime_settings().backend_url


BACKEND_URL = resolve_backend_url()


###############################################################################
@pytest.fixture(scope="session")
def base_url() -> str:
    # The backend serves the built Angular bundle on its own origin.
    return BACKEND_URL


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
###############################################################################
@pytest.fixture(scope="session")
def ml_api_context(playwright: Playwright, api_base_url: str) -> APIRequestContext:
    context = playwright.request.new_context(base_url=api_base_url)
    capabilities = context.get("/api/v1/system/capabilities")
    if not capabilities.ok or not capabilities.json().get("features", {}).get(
        "machine_learning"
    ):
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
