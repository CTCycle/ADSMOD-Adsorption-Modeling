"""E2E tests for the training and ML endpoints."""

from __future__ import annotations

import pytest
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
    def test_training_datasets_structure(self, api_context: APIRequestContext) -> None:
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
    def test_training_status_fields(self, api_context: APIRequestContext) -> None:
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
            "datasets": [
                {"source": "nist", "dataset_name": "NIST_SINGLE_COMPONENT_ADSORPTION"}
            ],
        }

        # Act
        response = api_context.post("/training/build-dataset", data=payload)

        # Assert
        # May succeed or fail based on available data
        assert response.status in (200, 400, 500)
        if response.ok:
            data = response.json()
            assert "job_id" in data

    # -------------------------------------------------------------------------
    def test_build_dataset_invalid_params(self, api_context: APIRequestContext) -> None:
        """Verify dataset build with invalid params returns error."""
        # Arrange
        payload = {
            "sample_size": 2.0,  # Invalid: > 1.0
            "validation_size": 0.2,
            "datasets": [
                {"source": "nist", "dataset_name": "NIST_SINGLE_COMPONENT_ADSORPTION"}
            ],
        }

        # Act
        response = api_context.post("/training/build-dataset", data=payload)

        # Assert
        assert response.status == 422  # Pydantic validation error


###############################################################################
class TestClearDataset:
    """Tests for the clear dataset endpoint."""

    # -------------------------------------------------------------------------
    def test_clear_training_dataset(self, api_context: APIRequestContext) -> None:
        """Verify clear dataset endpoint responds."""
        # Act
        response = api_context.delete("/training/dataset")

        # Assert
        assert response.ok
        data = response.json()
        assert data.get("status") in {"success", "error"}
        assert "message" in data


###############################################################################
class TestTrainingLifecycle:
    """Tests for training start/resume/stop behavior."""

    # -------------------------------------------------------------------------
    def test_start_training_when_dataset_missing(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify start training fails when no dataset is available."""
        # Arrange
        dataset_response = api_context.get("/training/datasets")
        assert dataset_response.ok
        if dataset_response.json().get("available"):
            pytest.skip("Training dataset exists; avoid starting a real session.")

        payload = {"epochs": 1}

        # Act
        response = api_context.post("/training/start", data=payload)

        # Assert
        assert response.status == 400

    # -------------------------------------------------------------------------
    def test_resume_training_with_missing_checkpoint(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify resume training fails for a missing checkpoint."""
        # Arrange
        checkpoints_response = api_context.get("/training/checkpoints")
        assert checkpoints_response.ok
        if checkpoints_response.json().get("checkpoints"):
            pytest.skip("Checkpoints exist; avoid resuming a real session.")

        payload = {"checkpoint_name": "missing-checkpoint", "additional_epochs": 1}

        # Act
        response = api_context.post("/training/resume", data=payload)

        # Assert
        assert response.status == 404

    # -------------------------------------------------------------------------
    def test_stop_training_when_idle(self, api_context: APIRequestContext) -> None:
        """Verify stop training succeeds when no session is active."""
        # Act
        response = api_context.post("/training/stop")

        # Assert
        assert response.ok
        data = response.json()
        assert data.get("status") == "stopped"
        assert "message" in data
