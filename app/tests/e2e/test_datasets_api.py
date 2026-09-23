"""E2E tests for the canonical dataset import and management endpoints."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from playwright.sync_api import APIRequestContext

###############################################################################
def _read_sample(sample_csv_path: str) -> bytes:
    with open(sample_csv_path, "rb") as handle:
        return handle.read()

###############################################################################
def _build_mapping(
    api_context: APIRequestContext,
    sample_csv_path: str,
    dataset_name: str,
) -> tuple[bytes, dict[str, Any]]:
    file_content = _read_sample(sample_csv_path)
    response = api_context.post(
        "/api/v1/datasets/import/preview",
        multipart={
            "file": {
                "name": f"{dataset_name}.csv",
                "mimeType": "text/csv",
                "buffer": file_content,
            }
        },
    )
    assert response.ok, f"Preview failed: {response.text()}"
    preview = response.json()
    column_roles = {
        column["name"]: column["proposed_role"] for column in preview["columns"]
    }
    grouping_columns = preview.get("proposed_grouping_columns") or [
        name
        for name, role in column_roles.items()
        if role in {"experiment_id", "experiment_name"}
    ]
    unit_overrides = {
        column["proposed_role"]: column["detected_unit"]
        for column in preview["columns"]
        if column.get("detected_unit")
        and column["proposed_role"] in {"pressure", "uptake", "temperature"}
    }
    mapping: dict[str, Any] = {
        "dataset_name": dataset_name,
        "structure": preview["detected_structure"],
        "column_roles": column_roles,
        "grouping_columns": grouping_columns,
        "pressure_basis": preview.get("proposed_pressure_basis") or "absolute",
        "unit_overrides": unit_overrides,
        "decimal_separator": ".",
        "duplicate_policy": "keep",
    }
    return file_content, mapping

###############################################################################
def _commit_sample(
    api_context: APIRequestContext,
    sample_csv_path: str,
    dataset_name: str,
) -> dict[str, Any]:
    file_content, mapping = _build_mapping(api_context, sample_csv_path, dataset_name)
    validation_response = api_context.post(
        "/api/v1/datasets/import/validate",
        multipart={
            "mapping": json.dumps(mapping),
            "file": {
                "name": f"{dataset_name}.csv",
                "mimeType": "text/csv",
                "buffer": file_content,
            },
        },
    )
    assert validation_response.ok, f"Validation failed: {validation_response.text()}"
    assert validation_response.json()["status"] == "valid"

    commit_response = api_context.post(
        "/api/v1/datasets/import/commit",
        multipart={
            "mapping": json.dumps(mapping),
            "file": {
                "name": f"{dataset_name}.csv",
                "mimeType": "text/csv",
                "buffer": file_content,
            },
        },
    )
    assert commit_response.ok, f"Commit failed: {commit_response.text()}"
    return commit_response.json()["dataset"]

###############################################################################
class TestDatasetImport:
    """Tests for the four-stage canonical dataset import flow."""

    # -------------------------------------------------------------------------
    def test_import_csv_dataset(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        dataset_name = f"test_adsorption_{uuid.uuid4().hex[:8]}"
        dataset = _commit_sample(api_context, sample_csv_path, dataset_name)

        assert dataset["name"] == dataset_name
        assert dataset["source"] == "uploaded"
        assert dataset["experiment_count"] > 0
        assert dataset["observation_count"] > 0

###############################################################################
class TestDatasetList:
    """Tests for listing canonical dataset summaries."""

    # -------------------------------------------------------------------------
    def test_get_dataset_list(self, api_context: APIRequestContext) -> None:
        response = api_context.get("/api/v1/datasets")

        assert response.ok, response.text()
        data = response.json()
        assert data["status"] == "success"
        assert isinstance(data["datasets"], list)

###############################################################################
class TestDatasetExperiments:
    """Tests for fetching experiments from an imported dataset."""

    # -------------------------------------------------------------------------
    def test_get_experiments_after_import(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        dataset = _commit_sample(
            api_context,
            sample_csv_path,
            f"experiment_test_{uuid.uuid4().hex[:8]}",
        )
        response = api_context.get(f"/api/v1/datasets/{dataset['id']}/experiments")

        assert response.ok, response.text()
        data = response.json()
        assert data["status"] == "success"
        assert len(data["experiments"]) > 0
        assert data["experiments"][0]["dataset_id"] == dataset["id"]

    # -------------------------------------------------------------------------
    def test_get_nonexistent_dataset_experiments(
        self, api_context: APIRequestContext
    ) -> None:
        response = api_context.get("/api/v1/datasets/999999/experiments")

        assert response.status == 404

###############################################################################
class TestDatasetDeletion:
    """Tests for deletion through the canonical dataset API."""

    # -------------------------------------------------------------------------
    def test_delete_uploaded_dataset_by_canonical_id(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        dataset = _commit_sample(
            api_context,
            sample_csv_path,
            f"delete_test_{uuid.uuid4().hex[:8]}",
        )

        response = api_context.delete(f"/api/v1/datasets/{dataset['id']}")

        assert response.status == 204
        listing = api_context.get("/api/v1/datasets")
        assert listing.ok
        assert dataset["id"] not in {item["id"] for item in listing.json()["datasets"]}

###############################################################################
class TestDatasetImportBoundaries:
    """Malformed input and incomplete mappings stay client-visible errors."""

    # -------------------------------------------------------------------------
    def test_preview_rejects_an_empty_upload(
        self, api_context: APIRequestContext
    ) -> None:
        response = api_context.post(
            "/api/v1/datasets/import/preview",
            multipart={
                "file": {
                    "name": "empty.csv",
                    "mimeType": "text/csv",
                    "buffer": b"",
                }
            },
        )

        assert response.status == 400
        assert response.json()["detail"] == "Uploaded dataset is empty."

    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("extension", [".xls", ".xlsx"])
    def test_preview_rejects_corrupt_excel_uploads(
        self,
        api_context: APIRequestContext,
        extension: str,
    ) -> None:
        response = api_context.post(
            "/api/v1/datasets/import/preview",
            multipart={
                "file": {
                    "name": f"corrupt{extension}",
                    "mimeType": "application/octet-stream",
                    "buffer": b"not an Excel workbook",
                }
            },
        )

        assert response.status == 400
        assert response.json()["detail"]

    # -------------------------------------------------------------------------
    def test_validation_reports_missing_required_columns(
        self,
        api_context: APIRequestContext,
    ) -> None:
        mapping = {
            "dataset_name": "missing-required-columns",
            "structure": "atomic",
            "column_roles": {
                "experiment_id": "experiment_id",
                "notes": "metadata",
            },
            "grouping_columns": ["experiment_id"],
            "pressure_basis": "absolute",
            "duplicate_policy": "keep",
        }
        response = api_context.post(
            "/api/v1/datasets/import/validate",
            multipart={
                "mapping": json.dumps(mapping),
                "file": {
                    "name": "missing-columns.csv",
                    "mimeType": "text/csv",
                    "buffer": b"experiment_id,notes\nEXP-1,source row\n",
                },
            },
        )

        assert response.ok, response.text()
        validation = response.json()
        assert validation["status"] == "invalid"
        issue_codes = {issue["code"] for issue in validation["issues"]}
        assert {"missing_pressure", "missing_uptake", "missing_temperature"} <= issue_codes
        assert {"missing_adsorbate", "missing_adsorbent"} <= issue_codes

    # -------------------------------------------------------------------------
    def test_validation_reports_malformed_numeric_values(
        self,
        api_context: APIRequestContext,
    ) -> None:
        mapping = {
            "dataset_name": "malformed-measurement",
            "structure": "atomic",
            "column_roles": {
                "experiment_id": "experiment_id",
                "pressure": "pressure",
                "uptake": "uptake",
                "temperature": "temperature",
                "adsorbate": "adsorbate",
                "adsorbent": "adsorbent",
            },
            "grouping_columns": ["experiment_id"],
            "pressure_basis": "absolute",
            "unit_overrides": {
                "pressure": "Pa",
                "uptake": "mol/kg",
                "temperature": "K",
            },
            "duplicate_policy": "keep",
        }
        response = api_context.post(
            "/api/v1/datasets/import/validate",
            multipart={
                "mapping": json.dumps(mapping),
                "file": {
                    "name": "malformed-measurement.csv",
                    "mimeType": "text/csv",
                    "buffer": (
                        b"experiment_id,pressure,uptake,temperature,adsorbate,adsorbent\n"
                        b"EXP-1,not-a-number,0.12,298.15,CO2,13X\n"
                    ),
                },
            },
        )

        assert response.ok, response.text()
        validation = response.json()
        assert validation["status"] == "invalid"
        assert "invalid_row" in {issue["code"] for issue in validation["issues"]}
