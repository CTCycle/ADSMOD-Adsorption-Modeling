# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""E2E tests for the dataset-level fitting session endpoints."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import APIRequestContext

from .test_datasets_api import _commit_sample

PRESSURES_BAR = (0.1, 0.3, 0.6, 1.0, 2.0, 4.0)


###############################################################################
def _write_fitting_csv(path: Path) -> str:
    """Write a two-experiment Langmuir-like dataset where every experiment is fittable."""
    lines = ["experiment,adsorbent,adsorbate,temperature [K],pressure [bar],uptake [mol/kg]"]
    for name, adsorbent, qsat in (("A", "Zeolite 13X", 4.0), ("B", "ZIF-8", 2.0)):
        for pressure in PRESSURES_BAR:
            uptake = qsat * pressure / (1 + pressure)
            lines.append(f"{name},{adsorbent},CO2,298.15,{pressure},{uptake:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


###############################################################################
def _wait_for_job(
    api_context: APIRequestContext, job_id: str, timeout_seconds: float = 60.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = api_context.get(f"/api/v1/fitting/jobs/{job_id}")
        assert response.ok, response.text()
        payload = response.json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.5)
    raise AssertionError(f"Job {job_id} did not finish within {timeout_seconds}s.")


###############################################################################
@pytest.fixture
def fitting_dataset_id(api_context: APIRequestContext, tmp_path: Path) -> int:
    csv_path = _write_fitting_csv(tmp_path / "fitting_sessions.csv")
    dataset = _commit_sample(api_context, csv_path, f"fitting_test_{uuid.uuid4().hex[:8]}")
    return int(dataset["id"])


###############################################################################
@pytest.fixture
def restore_model_settings(api_context: APIRequestContext) -> Iterator[None]:
    """Reset the global defaults of the models a test changes."""
    yield
    for key in ("langmuir", "freundlich"):
        response = api_context.post(f"/api/v1/fitting/models/{key}/reset")
        assert response.ok, response.text()
        response = api_context.put(
            f"/api/v1/fitting/models/{key}", data={"enabled": True, "parameters": {}}
        )
        assert response.ok, response.text()


###############################################################################
class TestFittingSessions:
    """Tests for starting and reading dataset-level fitting sessions."""

    # -------------------------------------------------------------------------
    def test_session_fits_every_experiment(
        self, api_context: APIRequestContext, fitting_dataset_id: int
    ) -> None:
        readiness = api_context.get(
            f"/api/v1/fitting/datasets/{fitting_dataset_id}/readiness"
        )
        assert readiness.ok, readiness.text()
        assert readiness.json()["ready"] is True

        started = api_context.post(
            "/api/v1/fitting/sessions",
            data={
                "name": "E2E session",
                "dataset_id": fitting_dataset_id,
                "models": ["langmuir", "freundlich"],
                "optimizer": "trf",
                "max_evaluations": 2000,
            },
        )
        assert started.ok, started.text()
        payload = started.json()

        job = _wait_for_job(api_context, payload["job_id"])
        assert job["status"] == "completed", job

        response = api_context.get(f"/api/v1/fitting/sessions/{payload['session_id']}")
        assert response.ok, response.text()
        session = response.json()
        assert session["name"] == "E2E session"
        assert session["status"] in {"completed", "warning"}
        assert sorted(item["experiment_name"] for item in session["experiments"]) == ["A", "B"]
        for experiment in session["experiments"]:
            assert {result["model"] for result in experiment["results"]} == {
                "langmuir",
                "freundlich",
            }
            assert experiment["best_model"] in {"langmuir", "freundlich"}

        listing = api_context.get(
            "/api/v1/fitting/sessions", params={"dataset_id": fitting_dataset_id}
        )
        assert listing.ok, listing.text()
        assert [item["id"] for item in listing.json()["items"]] == [payload["session_id"]]

    # -------------------------------------------------------------------------
    def test_disabled_model_is_rejected(
        self,
        api_context: APIRequestContext,
        fitting_dataset_id: int,
        restore_model_settings: None,
    ) -> None:
        disabled = api_context.put(
            "/api/v1/fitting/models/freundlich",
            data={"enabled": False, "parameters": {}},
        )
        assert disabled.ok, disabled.text()

        response = api_context.post(
            "/api/v1/fitting/sessions",
            data={
                "name": "Disabled model",
                "dataset_id": fitting_dataset_id,
                "models": ["langmuir", "freundlich"],
            },
        )

        assert response.status == 400
        assert "disabled" in response.json()["detail"]

    # -------------------------------------------------------------------------
    def test_invalid_optimizer_is_rejected(
        self, api_context: APIRequestContext, fitting_dataset_id: int
    ) -> None:
        response = api_context.post(
            "/api/v1/fitting/sessions",
            data={
                "name": "Bad optimizer",
                "dataset_id": fitting_dataset_id,
                "models": ["langmuir"],
                "optimizer": "INVALID_METHOD",
            },
        )

        assert response.status == 422


###############################################################################
class TestModelCatalog:
    """Tests for the fitting model catalog endpoint."""

    # -------------------------------------------------------------------------
    def test_get_model_catalog(self, api_context: APIRequestContext) -> None:
        response = api_context.get("/api/v1/fitting/models")

        assert response.ok, response.text()
        data = response.json()
        models = {model["key"]: model for model in data["models"]}
        assert {"langmuir", "freundlich"}.issubset(models)
        assert {
            parameter["name"] for parameter in models["langmuir"]["parameters"]
        } == {"k", "qsat"}


###############################################################################
class TestFittingJobs:
    """Tests for fitting job cancellation payloads."""

    # -------------------------------------------------------------------------
    def test_cancel_unknown_job_returns_error(
        self, api_context: APIRequestContext
    ) -> None:
        response = api_context.delete("/api/v1/fitting/jobs/unknown-job")
        assert response.status == 400
