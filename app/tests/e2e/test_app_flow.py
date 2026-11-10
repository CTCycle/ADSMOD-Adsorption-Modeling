# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""E2E tests for UI navigation and page rendering."""

from __future__ import annotations

from playwright.sync_api import Page, expect


###############################################################################
class TestHomepage:
    """Tests for the main application shell and custom datasets page."""

    # -------------------------------------------------------------------------
    def test_homepage_shows_shell_and_custom_dataset_controls(
        self, page: Page, base_url: str
    ) -> None:
        """Verify the homepage exposes its primary shell and upload controls."""
        page.goto(base_url)

        expect(page).to_have_title("Home · ADSMOD")
        expect(page.locator(".console-brand-name")).to_have_text("ADSMOD")
        expect(page.locator("nav[aria-label='Primary']").first).to_be_visible()

        page.get_by_role("navigation", name="Primary").get_by_role(
            "link", name="Datasets"
        ).click()
        expect(page.locator(".datasets-page")).to_be_visible()
        expect(page.locator("input[type='file']").first).to_be_attached()


###############################################################################
class TestHeaderNavigation:
    """Tests for header navigation between pages."""

    # -------------------------------------------------------------------------
    def test_navigate_to_models_page(self, page: Page, base_url: str) -> None:
        """Verify navigation to the fitting page exposes its model controls."""
        page.goto(base_url)
        page.get_by_role("navigation", name="Primary").get_by_role(
            "link", name="Modeling"
        ).click()
        expect(page.get_by_role("link", name="New session")).to_be_visible()

        page.get_by_role("navigation", name="Modeling sections").get_by_role(
            "link", name="Adsorption Models"
        ).click()
        expect(page.locator(".catalog-grid")).to_be_visible()
        expect(page.locator(".catalog-card").first).to_be_visible()
        expect(page.locator(".catalog-card[data-model='langmuir']")).to_be_visible()

    # -------------------------------------------------------------------------
    def test_navigate_to_training_page(self, page: Page, base_url: str) -> None:
        """Verify navigation to the Training page."""
        page.goto(base_url)
        page.get_by_role("navigation", name="Primary").get_by_role(
            "link", name="Modeling"
        ).click()
        page.get_by_role("navigation", name="Modeling sections").get_by_role(
            "link", name="Deep Learning"
        ).click()

        expect(page.locator(".training-page")).to_be_visible()
        data_processing = page.get_by_role("link", name="Data processing")
        expect(data_processing).to_be_visible()
        expect(data_processing).to_have_attribute("aria-current", "page")
