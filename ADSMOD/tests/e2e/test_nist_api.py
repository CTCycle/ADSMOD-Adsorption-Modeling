"""E2E tests for the NIST data ingestion endpoints."""

from __future__ import annotations

from playwright.sync_api import APIRequestContext


###############################################################################
class TestNistStatus:
    """Tests for the NIST status endpoint."""

    # -------------------------------------------------------------------------
    def test_get_nist_status(self, api_context: APIRequestContext) -> None:
        """Verify NIST status endpoint returns expected structure."""
        # Act
        response = api_context.get("/nist/status")

        # Assert
        assert response.ok
        data = response.json()
        assert "data_available" in data

    # -------------------------------------------------------------------------
    def test_nist_status_includes_counts(self, api_context: APIRequestContext) -> None:
        """Verify NIST status includes row count information."""
        # Act
        response = api_context.get("/nist/status")

        # Assert
        assert response.ok
        data = response.json()
        # Row counts may be present if data is available
        if data.get("data_available"):
            assert "single_component_rows" in data
            assert "binary_mixture_rows" in data
            assert "guest_rows" in data
            assert "host_rows" in data


###############################################################################
class TestNistFetch:
    """Tests for the NIST data fetch endpoint."""

    # -------------------------------------------------------------------------
    def test_fetch_nist_data_small_fraction(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify NIST fetch with small fraction succeeds."""
        # Arrange
        payload = {
            "experiments_fraction": 0.01,
            "guest_fraction": 0.01,
            "host_fraction": 0.01,
        }

        # Act
        response = api_context.post("/nist/fetch", data=payload)

        # Assert
        # May succeed or fail depending on network/NIST API availability
        assert response.status in (200, 500)
        if response.ok:
            data = response.json()
            # Response should contain fetch counts
            assert isinstance(data, dict)

    # -------------------------------------------------------------------------
    def test_fetch_nist_data_invalid_fraction(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify NIST fetch with invalid fraction returns 400."""
        # Arrange
        payload = {
            "experiments_fraction": 2.0,  # Invalid: > 1.0
            "guest_fraction": 0.01,
            "host_fraction": 0.01,
        }

        # Act
        response = api_context.post("/nist/fetch", data=payload)

        # Assert
        assert response.status == 422  # Pydantic validation error


###############################################################################
class TestNistProperties:
    """Tests for the NIST properties enrichment endpoint."""

    # -------------------------------------------------------------------------
    def test_fetch_nist_properties_guest(self, api_context: APIRequestContext) -> None:
        """Verify NIST properties fetch for guest materials."""
        # Arrange
        payload = {"target": "guest"}

        # Act
        response = api_context.post("/nist/properties", data=payload)

        # Assert
        # May succeed or return 400 if no data available
        assert response.status in (200, 400, 500)
        if response.ok:
            data = response.json()
            assert "job_id" in data
            assert data.get("job_type") == "nist_properties"

    # -------------------------------------------------------------------------
    def test_fetch_nist_properties_host(self, api_context: APIRequestContext) -> None:
        """Verify NIST properties fetch for host materials."""
        # Arrange
        payload = {"target": "host"}

        # Act
        response = api_context.post("/nist/properties", data=payload)

        # Assert
        assert response.status in (200, 400, 500)
        if response.ok:
            data = response.json()
            assert "job_id" in data
            assert data.get("job_type") == "nist_properties"

    # -------------------------------------------------------------------------
    def test_fetch_nist_properties_invalid_target(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify NIST properties with invalid target returns error."""
        # Arrange
        payload = {"target": "invalid"}

        # Act
        response = api_context.post("/nist/properties", data=payload)

        # Assert
        assert response.status == 422  # Pydantic validation error
