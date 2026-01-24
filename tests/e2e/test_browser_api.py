"""E2E tests for the database browser endpoints."""

from __future__ import annotations

from playwright.sync_api import APIRequestContext


###############################################################################
class TestListTables:
    """Tests for the table listing endpoint."""

    # -------------------------------------------------------------------------
    def test_list_tables(self, api_context: APIRequestContext) -> None:
        """Verify tables endpoint returns list of available tables."""
        # Act
        response = api_context.get("/browser/tables")

        # Assert
        assert response.ok
        data = response.json()
        assert "tables" in data
        assert isinstance(data["tables"], list)
        assert len(data["tables"]) > 0

    # -------------------------------------------------------------------------
    def test_tables_have_required_fields(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify each table entry has required fields."""
        # Act
        response = api_context.get("/browser/tables")

        # Assert
        assert response.ok
        data = response.json()
        for table in data["tables"]:
            assert "table_name" in table
            assert "display_name" in table
            assert "category" in table

    # -------------------------------------------------------------------------
    def test_tables_include_adsorption_data(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify ADSORPTION_DATA table is in the list."""
        # Act
        response = api_context.get("/browser/tables")

        # Assert
        assert response.ok
        data = response.json()
        table_names = [t["table_name"] for t in data["tables"]]
        assert "ADSORPTION_DATA" in table_names


###############################################################################
class TestGetTableData:
    """Tests for fetching table data."""

    # -------------------------------------------------------------------------
    def test_get_table_data_adsorption(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify fetching ADSORPTION_DATA table succeeds."""
        # Act
        response = api_context.get("/browser/data/ADSORPTION_DATA")

        # Assert
        assert response.ok
        data = response.json()
        assert "table_name" in data
        assert data["table_name"] == "ADSORPTION_DATA"
        assert "columns" in data
        assert "data" in data
        assert "row_count" in data
        assert "column_count" in data

    # -------------------------------------------------------------------------
    def test_get_table_data_langmuir(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify fetching ADSORPTION_LANGMUIR table succeeds."""
        # Act
        response = api_context.get("/browser/data/ADSORPTION_LANGMUIR")

        # Assert
        assert response.ok
        data = response.json()
        assert data["table_name"] == "ADSORPTION_LANGMUIR"

    # -------------------------------------------------------------------------
    def test_get_invalid_table(self, api_context: APIRequestContext) -> None:
        """Verify 404 error when fetching a non-existent table."""
        # Act
        response = api_context.get("/browser/data/INVALID_TABLE_NAME")

        # Assert
        assert response.status == 404


###############################################################################
class TestTableCategories:
    """Tests for table category groupings."""

    # -------------------------------------------------------------------------
    def test_tables_grouped_by_category(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify tables are properly categorized."""
        # Act
        response = api_context.get("/browser/tables")

        # Assert
        assert response.ok
        data = response.json()
        categories = set(t["category"] for t in data["tables"])

        # Should have at least some expected categories
        expected_categories = {"NIST-A Data", "Uploaded Data", "Model Results"}
        assert len(categories & expected_categories) > 0
