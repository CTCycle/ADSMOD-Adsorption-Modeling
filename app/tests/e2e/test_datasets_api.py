"""E2E tests for dataset upload and management endpoints."""

from __future__ import annotations

from playwright.sync_api import APIRequestContext


###############################################################################
class TestDatasetUpload:
    """Tests for the dataset upload endpoint."""

    # -------------------------------------------------------------------------
    def test_upload_csv_dataset(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        """Verify CSV upload returns the complete dataset response contract."""
        with open(sample_csv_path, "rb") as f:
            file_content = f.read()

        response = api_context.post(
            "/api/datasets/load",
            multipart={
                "file": {
                    "name": "test_adsorption.csv",
                    "mimeType": "text/csv",
                    "buffer": file_content,
                }
            },
        )

        assert response.ok, f"Upload failed: {response.text()}"
        data = response.json()
        assert "summary" in data
        assert "dataset" in data
        assert data["dataset"]["dataset_name"] == "test_adsorption"
        assert len(data["dataset"]["records"]) > 0
        assert len(data["dataset"]["columns"]) >= 4


###############################################################################
class TestDatasetNames:
    """Tests for the dataset names listing endpoint."""

    # -------------------------------------------------------------------------
    def test_get_dataset_names(self, api_context: APIRequestContext) -> None:
        """Verify dataset names endpoint returns a list."""
        response = api_context.get("/api/datasets/names")

        assert response.ok
        data = response.json()
        assert "names" in data
        assert isinstance(data["names"], list)


###############################################################################
class TestDatasetByName:
    """Tests for fetching datasets by name."""

    # -------------------------------------------------------------------------
    def test_get_dataset_by_name_after_upload(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        """Verify a dataset can be fetched by name after upload."""
        with open(sample_csv_path, "rb") as f:
            file_content = f.read()

        upload_response = api_context.post(
            "/api/datasets/load",
            multipart={
                "file": {
                    "name": "fetch_test.csv",
                    "mimeType": "text/csv",
                    "buffer": file_content,
                }
            },
        )
        assert upload_response.ok

        response = api_context.get("/api/datasets/by-name/fetch_test")

        assert response.ok
        data = response.json()
        assert "dataset" in data
        assert data["dataset"]["dataset_name"] == "fetch_test"

    # -------------------------------------------------------------------------
    def test_get_nonexistent_dataset(self, api_context: APIRequestContext) -> None:
        """Verify fetching a non-existent dataset returns a client error."""
        response = api_context.get("/api/datasets/by-name/nonexistent_dataset_xyz")

        assert response.status == 400
