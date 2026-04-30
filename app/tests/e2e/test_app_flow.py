"""E2E tests for UI navigation and page rendering."""

from __future__ import annotations

from playwright.sync_api import Page, expect


###############################################################################
class TestHomepage:
    """Tests for the main application homepage."""

    # -------------------------------------------------------------------------
    def test_homepage_loads(self, page: Page, base_url: str) -> None:
        """Verify the application homepage loads with correct title."""
        # Arrange & Act
        page.goto(base_url)

        # Assert
        expect(page).to_have_title("ADSMOD Adsorption Modeling")

    # -------------------------------------------------------------------------
    def test_header_displays(self, page: Page, base_url: str) -> None:
        """Verify the application header is visible."""
        # Arrange & Act
        page.goto(base_url)

        # Assert
        header = page.locator(".app-header h1")
        expect(header).to_be_visible()
        expect(header).to_have_text("ADSMOD")


###############################################################################
class TestHeaderNavigation:
    """Tests for header navigation between pages."""

    # -------------------------------------------------------------------------
    def test_navigation_visible(self, page: Page, base_url: str) -> None:
        """Verify the main navigation is visible on page load."""
        # Arrange & Act
        page.goto(base_url)

        # Assert
        navigation = page.get_by_label("Main navigation")
        expect(navigation).to_be_visible()

    # -------------------------------------------------------------------------
    def test_navigate_to_models_page(self, page: Page, base_url: str) -> None:
        """Verify navigation to the Models page."""
        # Arrange
        page.goto(base_url)

        # Act
        page.get_by_title("Fitting").click()

        # Assert
        models_section = page.locator("section:not([hidden]) .models-grid")
        expect(models_section).to_be_visible()

    # -------------------------------------------------------------------------
    def test_navigate_to_training_page(self, page: Page, base_url: str) -> None:
        """Verify navigation to the Training page."""
        # Arrange
        page.goto(base_url)

        # Act
        page.get_by_title("Training").click()

        # Assert
        training_section = page.locator("section:not([hidden]) .ml-page")
        expect(training_section).to_be_visible()


###############################################################################
class TestConfigPage:
    """Tests for the Fitting Configuration page elements."""

    # -------------------------------------------------------------------------
    def test_config_page_default(self, page: Page, base_url: str) -> None:
        """Verify the Config page is shown by default."""
        # Arrange & Act
        page.goto(base_url)

        # Assert
        config_section = page.locator(".source-page")
        expect(config_section).to_be_visible()

    # -------------------------------------------------------------------------
    def test_file_upload_area_visible(self, page: Page, base_url: str) -> None:
        """Verify the file upload area is present."""
        # Arrange & Act
        page.goto(base_url)

        # Assert
        upload_input = page.locator("input[type='file']").first
        expect(upload_input).to_be_attached()


###############################################################################
class TestModelsPage:
    """Tests for the Models page elements."""

    # -------------------------------------------------------------------------
    def test_model_cards_displayed(self, page: Page, base_url: str) -> None:
        """Verify model cards are displayed on the Models page."""
        # Arrange
        page.goto(base_url)

        # Act
        page.get_by_title("Fitting").click()

        # Assert
        model_cards = page.locator(".model-grid-card")
        expect(model_cards.first).to_be_visible()

    # -------------------------------------------------------------------------
    def test_langmuir_model_present(self, page: Page, base_url: str) -> None:
        """Verify the Langmuir model card is present."""
        # Arrange
        page.goto(base_url)

        # Act
        page.get_by_title("Fitting").click()

        # Assert
        langmuir_card = page.locator("#model-card-langmuir")
        expect(langmuir_card).to_be_visible()

    # -------------------------------------------------------------------------
    def test_fitting_controls_visible(self, page: Page, base_url: str) -> None:
        """Verify fitting controls (iterations, method) are visible."""
        # Arrange
        page.goto(base_url)

        # Act
        page.get_by_title("Fitting").click()

        # Assert
        start_button = page.locator("button:has-text('Start Fitting')")
        expect(start_button).to_be_visible()

