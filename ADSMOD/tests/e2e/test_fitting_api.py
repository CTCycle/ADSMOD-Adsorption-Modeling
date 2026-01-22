"""E2E tests for the fitting pipeline endpoints."""

from __future__ import annotations

from playwright.sync_api import APIRequestContext


###############################################################################
class TestFittingRun:
    """Tests for the fitting run endpoint."""

    # -------------------------------------------------------------------------
    def test_run_fitting_langmuir(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        """Verify fitting with Langmuir model succeeds."""
        # Arrange - first upload a dataset
        with open(sample_csv_path, "rb") as f:
            file_content = f.read()

        upload_response = api_context.post(
            "/datasets/load",
            multipart={
                "file": {
                    "name": "fitting_test.csv",
                    "mimeType": "text/csv",
                    "buffer": file_content,
                }
            },
        )
        assert upload_response.ok
        dataset = upload_response.json()["dataset"]

        # Build fitting request
        payload = {
            "dataset": dataset,
            "parameter_bounds": {
                "Langmuir": {
                    "min": {"k": 1e-06, "qsat": 0.0},
                    "max": {"k": 10.0, "qsat": 100.0},
                    "initial": {"k": 0.5, "qsat": 50.0},
                }
            },
            "max_iterations": 1000,
            "optimization_method": "LSS",
        }

        # Act
        response = api_context.post("/fitting/run", data=payload)

        # Assert
        assert response.ok, f"Fitting failed: {response.text()}"
        data = response.json()
        assert "summary" in data or "message" in data

    # -------------------------------------------------------------------------
    def test_run_fitting_multiple_models(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        """Verify fitting with multiple models succeeds."""
        # Arrange
        with open(sample_csv_path, "rb") as f:
            file_content = f.read()

        upload_response = api_context.post(
            "/datasets/load",
            multipart={
                "file": {
                    "name": "multi_model_test.csv",
                    "mimeType": "text/csv",
                    "buffer": file_content,
                }
            },
        )
        assert upload_response.ok
        dataset = upload_response.json()["dataset"]

        payload = {
            "dataset": dataset,
            "parameter_bounds": {
                "Langmuir": {
                    "min": {"k": 1e-06, "qsat": 0.0},
                    "max": {"k": 10.0, "qsat": 100.0},
                    "initial": {"k": 0.5, "qsat": 50.0},
                },
                "Freundlich": {
                    "min": {"k": 1e-06, "exponent": 0.1},
                    "max": {"k": 10.0, "exponent": 10.0},
                    "initial": {"k": 0.5, "exponent": 1.0},
                },
            },
            "max_iterations": 500,
            "optimization_method": "LSS",
        }

        # Act
        response = api_context.post("/fitting/run", data=payload)

        # Assert
        assert response.ok

    # -------------------------------------------------------------------------
    def test_run_fitting_invalid_method(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        """Verify fitting with invalid optimization method returns error."""
        # Arrange
        with open(sample_csv_path, "rb") as f:
            file_content = f.read()

        upload_response = api_context.post(
            "/datasets/load",
            multipart={
                "file": {
                    "name": "invalid_method_test.csv",
                    "mimeType": "text/csv",
                    "buffer": file_content,
                }
            },
        )
        assert upload_response.ok
        dataset = upload_response.json()["dataset"]

        payload = {
            "dataset": dataset,
            "parameter_bounds": {
                "Langmuir": {
                    "min": {"k": 1e-06, "qsat": 0.0},
                    "max": {"k": 10.0, "qsat": 100.0},
                    "initial": {"k": 0.5, "qsat": 50.0},
                }
            },
            "max_iterations": 100,
            "optimization_method": "INVALID_METHOD",
        }

        # Act
        response = api_context.post("/fitting/run", data=payload)

        # Assert
        assert response.status == 422  # Pydantic validation error


###############################################################################
class TestNistDatasetForFitting:
    """Tests for the NIST dataset endpoint."""

    # -------------------------------------------------------------------------
    def test_get_nist_dataset_for_fitting(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify NIST dataset endpoint returns data when available."""
        # Act
        response = api_context.get("/fitting/nist-dataset")

        # Assert
        # May return 200 with data or 400 if no NIST data available
        assert response.status in (200, 400)
        if response.ok:
            data = response.json()
            assert "dataset" in data or "summary" in data
