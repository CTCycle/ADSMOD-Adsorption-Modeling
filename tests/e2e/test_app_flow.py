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
        expect(header).to_have_text("ADSMOD Adsorption Modeling")


###############################################################################
class TestSidebarNavigation:
    """Tests for sidebar navigation between pages."""

    # -------------------------------------------------------------------------
    def test_sidebar_visible(self, page: Page, base_url: str) -> None:
        """Verify the sidebar is visible on page load."""
        # Arrange & Act
        page.goto(base_url)

        # Assert
        sidebar = page.locator(".sidebar")
        expect(sidebar).to_be_visible()

    # -------------------------------------------------------------------------
    def test_navigate_to_models_page(self, page: Page, base_url: str) -> None:
        """Verify navigation to the Models page."""
        # Arrange
        page.goto(base_url)

        # Act
        page.get_by_title("Models & Fitting").click()

        # Assert
        models_section = page.locator("section:not([hidden]) .models-grid")
        expect(models_section).to_be_visible()

    # -------------------------------------------------------------------------
    def test_navigate_to_browser_page(self, page: Page, base_url: str) -> None:
        """Verify navigation to the Database Browser page."""
        # Arrange
        page.goto(base_url)

        # Act
        page.get_by_title("Database Browser").click()

        # Assert
        browser_section = page.locator("section:not([hidden]) .browser-page")
        expect(browser_section).to_be_visible()

    # -------------------------------------------------------------------------
    def test_navigate_to_analysis_page(self, page: Page, base_url: str) -> None:
        """Verify navigation to the Analysis (ML) page."""
        # Arrange
        page.goto(base_url)

        # Act
        page.get_by_title("Analysis").click()

        # Assert
        analysis_section = page.locator("section:not([hidden]) .ml-page")
        expect(analysis_section).to_be_visible()


###############################################################################
class TestConfigPage:
    """Tests for the Fitting Configuration page elements."""

    # -------------------------------------------------------------------------
    def test_config_page_default(self, page: Page, base_url: str) -> None:
        """Verify the Config page is shown by default."""
        # Arrange & Act
        page.goto(base_url)

        # Assert
        config_section = page.locator("section:not([hidden])")
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
        page.get_by_title("Models & Fitting").click()

        # Assert
        model_cards = page.locator(".model-grid-card")
        expect(model_cards.first).to_be_visible()

    # -------------------------------------------------------------------------
    def test_langmuir_model_present(self, page: Page, base_url: str) -> None:
        """Verify the Langmuir model card is present."""
        # Arrange
        page.goto(base_url)

        # Act
        page.get_by_title("Models & Fitting").click()

        # Assert
        langmuir_card = page.locator("#model-card-langmuir")
        expect(langmuir_card).to_be_visible()

    # -------------------------------------------------------------------------
    def test_fitting_controls_visible(self, page: Page, base_url: str) -> None:
        """Verify fitting controls (iterations, method) are visible."""
        # Arrange
        page.goto(base_url)

        # Act
        page.get_by_title("Models & Fitting").click()

        # Assert
        start_button = page.locator("button:has-text('Start Fitting')")
        expect(start_button).to_be_visible()


###############################################################################
class TestDatabaseBrowserPage:
    """Tests for the Database Browser page elements."""

    # -------------------------------------------------------------------------
    def test_table_dropdown_visible(self, page: Page, base_url: str) -> None:
        """Verify the table selection dropdown is visible."""
        # Arrange
        page.goto(base_url)

        # Act
        page.get_by_title("Database Browser").click()

        # Assert
        dropdown = page.locator("select").first
        expect(dropdown).to_be_visible()
