"""E2E tests for retired database browser endpoints."""

from __future__ import annotations

from playwright.sync_api import APIRequestContext


###############################################################################
class TestListTables:
    """Tests for the retired table listing endpoint."""

    # -------------------------------------------------------------------------
    def test_list_tables(self, api_context: APIRequestContext) -> None:
        """Verify /browser/tables is no longer available."""
        # Act
        response = api_context.get("/api/browser/tables")

        # Assert
        assert response.status == 404

    # -------------------------------------------------------------------------
    def test_tables_have_required_fields(self, api_context: APIRequestContext) -> None:
        """Verify /browser/tables no longer returns table metadata."""
        # Act
        response = api_context.get("/api/browser/tables")

        # Assert
        assert response.status == 404

    # -------------------------------------------------------------------------
    def test_tables_include_adsorption_data(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify /browser/tables does not expose adsorption_data anymore."""
        # Act
        response = api_context.get("/api/browser/tables")

        # Assert
        assert response.status == 404


###############################################################################
class TestGetTableData:
    """Tests for retired table data endpoints."""

    # -------------------------------------------------------------------------
    def test_get_table_data_adsorption(self, api_context: APIRequestContext) -> None:
        """Verify /browser/data/adsorption_data is no longer available."""
        # Act
        response = api_context.get("/api/browser/data/adsorption_data")

        # Assert
        assert response.status == 404

    # -------------------------------------------------------------------------
    def test_get_table_data_langmuir(self, api_context: APIRequestContext) -> None:
        """Verify /browser/data/adsorption_langmuir is no longer available."""
        # Act
        response = api_context.get("/api/browser/data/adsorption_langmuir")

        # Assert
        assert response.status == 404

    # -------------------------------------------------------------------------
    def test_get_invalid_table(self, api_context: APIRequestContext) -> None:
        """Verify invalid browser table path returns 404."""
        # Act
        response = api_context.get("/api/browser/data/INVALID_TABLE_NAME")

        # Assert
        assert response.status == 404


###############################################################################
class TestTableCategories:
    """Tests for retired table category endpoint behavior."""

    # -------------------------------------------------------------------------
    def test_tables_grouped_by_category(self, api_context: APIRequestContext) -> None:
        """Verify /browser/tables no longer returns categories."""
        # Act
        response = api_context.get("/api/browser/tables")

        # Assert
        assert response.status == 404
