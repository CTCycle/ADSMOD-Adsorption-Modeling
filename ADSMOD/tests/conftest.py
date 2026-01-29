"""Pytest configuration and shared fixtures for ADSMOD E2E tests."""

from __future__ import annotations

import os

import pytest
from playwright.sync_api import APIRequestContext, Page, Playwright


# [CONSTANTS]
###############################################################################
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(TESTS_DIR, "fixtures")
FRONTEND_URL = "http://127.0.0.1:7861"
BACKEND_URL = "http://127.0.0.1:8000"


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
