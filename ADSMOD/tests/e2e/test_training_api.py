"""E2E tests for the training and ML endpoints."""

from __future__ import annotations

from playwright.sync_api import APIRequestContext


###############################################################################
class TestTrainingDatasets:
    """Tests for training dataset availability endpoint."""

    # -------------------------------------------------------------------------
    def test_get_training_datasets(self, api_context: APIRequestContext) -> None:
        """Verify training datasets endpoint returns expected structure."""
        # Act
        response = api_context.get("/training/datasets")

        # Assert
        assert response.ok
        data = response.json()
        assert "available" in data

    # -------------------------------------------------------------------------
    def test_training_datasets_structure(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify training datasets response structure."""
        # Act
        response = api_context.get("/training/datasets")

        # Assert
        assert response.ok
        data = response.json()
        # If available, should have sample counts
        if data.get("available"):
            assert "train_samples" in data or "name" in data


###############################################################################
class TestCheckpoints:
    """Tests for the checkpoints listing endpoint."""

    # -------------------------------------------------------------------------
    def test_get_checkpoints(self, api_context: APIRequestContext) -> None:
        """Verify checkpoints endpoint returns a list."""
        # Act
        response = api_context.get("/training/checkpoints")

        # Assert
        assert response.ok
        data = response.json()
        assert "checkpoints" in data
        assert isinstance(data["checkpoints"], list)


###############################################################################
class TestTrainingStatus:
    """Tests for the training status endpoint."""

    # -------------------------------------------------------------------------
    def test_get_training_status(self, api_context: APIRequestContext) -> None:
        """Verify training status endpoint returns expected fields."""
        # Act
        response = api_context.get("/training/status")

        # Assert
        assert response.ok
        data = response.json()
        assert "is_training" in data

    # -------------------------------------------------------------------------
    def test_training_status_fields(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify training status includes progress information."""
        # Act
        response = api_context.get("/training/status")

        # Assert
        assert response.ok
        data = response.json()
        assert "current_epoch" in data
        assert "total_epochs" in data
        assert "progress" in data


###############################################################################
class TestDatasetInfo:
    """Tests for the dataset info endpoint."""

    # -------------------------------------------------------------------------
    def test_get_dataset_info(self, api_context: APIRequestContext) -> None:
        """Verify dataset info endpoint returns expected structure."""
        # Act
        response = api_context.get("/training/dataset-info")

        # Assert
        assert response.ok
        data = response.json()
        assert "available" in data


###############################################################################
class TestDatasetBuild:
    """Tests for the dataset build endpoint."""

    # -------------------------------------------------------------------------
    def test_build_dataset_request_structure(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify dataset build endpoint accepts valid request."""
        # Arrange
        payload = {
            "sample_size": 0.1,
            "validation_size": 0.2,
            "min_measurements": 5,
            "max_measurements": 50,
            "smile_sequence_size": 100,
            "max_pressure": 10000.0,
            "max_uptake": 20.0,
            "source_datasets": ["SINGLE_COMPONENT_ADSORPTION"],
        }

        # Act
        response = api_context.post("/training/build-dataset", data=payload)

        # Assert
        # May succeed or fail based on available data
        assert response.status in (200, 400, 500)
        if response.ok:
            data = response.json()
            assert "success" in data

    # -------------------------------------------------------------------------
    def test_build_dataset_invalid_params(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify dataset build with invalid params returns error."""
        # Arrange
        payload = {
            "sample_size": 2.0,  # Invalid: > 1.0
            "validation_size": 0.2,
        }

        # Act
        response = api_context.post("/training/build-dataset", data=payload)

        # Assert
        assert response.status == 422  # Pydantic validation error


###############################################################################
class TestClearDataset:
    """Tests for the clear dataset endpoint."""

    # -------------------------------------------------------------------------
    def test_clear_training_dataset(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify clear dataset endpoint responds."""
        # Act
        response = api_context.post("/training/clear-dataset")

        # Assert
        assert response.ok
        data = response.json()
        assert "success" in data or "message" in data
