"""E2E tests for UI navigation and page rendering."""

from __future__ import annotations

from playwright.sync_api import Page, expect


###############################################################################
class TestHomepage:
    """Tests for the main application shell and source page."""

    # -------------------------------------------------------------------------
    def test_homepage_shows_shell_and_source_controls(
        self, page: Page, base_url: str
    ) -> None:
        """Verify the homepage exposes its primary shell and source controls."""
        page.goto(base_url)

        expect(page).to_have_title("ADSMOD Adsorption Modeling")
        expect(page.locator(".app-header h1")).to_have_text("ADSMOD")
        expect(page.get_by_label("Main navigation")).to_be_visible()
        expect(page.locator(".source-page")).to_be_visible()
        expect(page.locator("input[type='file']").first).to_be_attached()


###############################################################################
class TestHeaderNavigation:
    """Tests for header navigation between pages."""

    # -------------------------------------------------------------------------
    def test_navigate_to_models_page(self, page: Page, base_url: str) -> None:
        """Verify navigation to the fitting page exposes its model controls."""
        page.goto(base_url)
        page.get_by_title("Fitting").click()

        expect(page.locator("section:not([hidden]) .models-grid")).to_be_visible()
        expect(page.locator(".model-grid-card").first).to_be_visible()
        expect(page.locator("#model-card-langmuir")).to_be_visible()
        expect(page.locator("button:has-text('Start Fitting')")).to_be_visible()

    # -------------------------------------------------------------------------
    def test_navigate_to_training_page(self, page: Page, base_url: str) -> None:
        """Verify navigation to the Training page."""
        page.goto(base_url)
        page.get_by_title("Training").click()

        expect(page.locator("section:not([hidden]) .ml-page")).to_be_visible()
