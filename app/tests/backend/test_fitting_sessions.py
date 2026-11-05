# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Global adsorption model defaults and dataset-level fitting sessions."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings

CSV = (
    "experiment,adsorbent,adsorbate,temperature [K],pressure [bar],uptake [mol/kg]\n"
    + "".join(
        f"{name},{adsorbent},CO2,298.15,{p},{q * p / (1 + p):.6f}\n"
        for name, adsorbent, q in (("A", "Zeolite 13X", 4.0), ("B", "ZIF-8", 2.0))
        for p in (0.1, 0.3, 0.6, 1.0, 2.0, 4.0)
    )
)
MAPPING: dict[str, Any] = {
    "structure": "atomic",
    "column_roles": {
        "experiment": "experiment_name",
        "adsorbent": "adsorbent",
        "adsorbate": "adsorbate",
        "temperature [K]": "temperature",
        "pressure [bar]": "pressure",
        "uptake [mol/kg]": "uptake",
    },
    "grouping_columns": ["experiment"],
    "unit_overrides": {"pressure": "bar", "uptake": "mol/kg", "temperature": "K"},
    "pressure_basis": "absolute",
    "duplicate_policy": "reject",
}


###############################################################################
@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    config = AdsmodConfig.assemble(
        database=DatabaseConfig(embedded_database=True, sqlite_path="data/test.db"),
        storage_root=tmp_path,
        settings=UserSettings(),
    )
    with TestClient(create_app(config)) as test_client:
        yield test_client


###############################################################################
def _import(client: TestClient, name: str = "Two zeolites") -> int:
    mapping = {**MAPPING, "dataset_name": name}
    files = {"file": ("fit.csv", CSV.encode("utf-8"), "text/csv")}
    validation = client.post(
        "/api/v1/datasets/import/validate",
        data={"mapping": json.dumps(mapping)},
        files=files,
    ).json()
    mapping["confirmed_issue_codes"] = sorted(
        {issue["code"] for issue in validation["issues"] if issue["severity"] == "confirmation"}
    )
    response = client.post(
        "/api/v1/datasets/import/commit",
        data={"mapping": json.dumps(mapping)},
        files=files,
    )
    assert response.status_code == 201, response.text
    return int(response.json()["dataset"]["id"])


###############################################################################
def _model(client: TestClient, key: str) -> dict[str, Any]:
    models = client.get("/api/v1/fitting/models").json()["models"]
    return next(model for model in models if model["key"] == key)


###############################################################################
def _run(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/api/v1/fitting/sessions", json=payload)
    assert response.status_code == 200, response.text
    started = response.json()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/fitting/jobs/{started['job_id']}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    detail = client.get(f"/api/v1/fitting/sessions/{started['session_id']}")
    assert detail.status_code == 200, detail.text
    return detail.json()


###############################################################################
def test_global_model_defaults_validate_persist_and_restore(client: TestClient) -> None:
    langmuir = _model(client, "langmuir")
    assert langmuir["enabled"] is True
    assert all(not parameter["customized"] for parameter in langmuir["parameters"])

    invalid = client.put(
        "/api/v1/fitting/models/langmuir",
        json={"enabled": True, "parameters": {"k": {"lower": 0.0, "upper": 1.0}}},
    )
    assert invalid.status_code == 400
    inverted = client.put(
        "/api/v1/fitting/models/langmuir",
        json={"enabled": True, "parameters": {"k": {"lower": 2.0, "upper": 1.0}}},
    )
    assert inverted.status_code == 422
    unknown = client.put(
        "/api/v1/fitting/models/langmuir",
        json={"enabled": True, "parameters": {"zeta": {"lower": 1.0, "upper": 2.0}}},
    )
    assert unknown.status_code == 400
    assert _model(client, "langmuir") == langmuir

    saved = client.put(
        "/api/v1/fitting/models/langmuir",
        json={"enabled": False, "parameters": {"k": {"lower": 1e-9, "upper": 1e-3}}},
    )
    assert saved.status_code == 200, saved.text
    stored = _model(client, "langmuir")
    k = next(parameter for parameter in stored["parameters"] if parameter["name"] == "k")
    assert stored["enabled"] is False
    assert (k["lower"], k["upper"], k["customized"]) == (1e-9, 1e-3, True)

    restored = client.post("/api/v1/fitting/models/langmuir/reset").json()
    assert restored["enabled"] is False
    assert restored["parameters"] == langmuir["parameters"]


###############################################################################
def test_session_fits_every_experiment_with_snapshot_and_lifecycle(
    client: TestClient,
) -> None:
    dataset_id = _import(client)
    readiness = client.get(f"/api/v1/fitting/datasets/{dataset_id}/readiness").json()
    assert readiness["ready"] is True
    assert readiness["experiment_count"] == 2
    assert readiness["inverse_sigma_available"] is False
    assert "dubinin_radushkevich" in readiness["model_incompatibility"]

    client.put(
        "/api/v1/fitting/models/langmuir",
        json={"enabled": True, "parameters": {"qsat": {"lower": 0.5, "upper": 50.0}}},
    )
    client.put("/api/v1/fitting/models/sips", json={"enabled": False, "parameters": {}})
    base = {"dataset_id": dataset_id, "max_evaluations": 2000}

    disabled = client.post(
        "/api/v1/fitting/sessions",
        json={**base, "name": "Disabled", "models": ["langmuir", "sips"]},
    )
    assert disabled.status_code == 400
    assert "disabled" in disabled.json()["detail"]
    incompatible = client.post(
        "/api/v1/fitting/sessions",
        json={**base, "name": "DR", "models": ["dubinin_radushkevich"]},
    )
    assert incompatible.status_code == 400

    detail = _run(
        client,
        {
            **base,
            "name": "  Zeolite screening  ",
            "models": ["langmuir", "toth"],
            "parameter_overrides": {"toth": {"t": {"lower": 0.2, "upper": 3.0}}},
        },
    )
    assert detail["name"] == "Zeolite screening"
    assert detail["status"] in {"completed", "warning"}
    assert detail["experiment_count"] == 2
    assert sorted(item["experiment_name"] for item in detail["experiments"]) == ["A", "B"]
    for experiment in detail["experiments"]:
        assert {result["model"] for result in experiment["results"]} == {"langmuir", "toth"}
        assert experiment["best_model"] in {"langmuir", "toth"}
    snapshot = detail["configuration"]["model_bounds"]
    assert snapshot["langmuir"]["qsat"] == {"lower": 0.5, "upper": 50.0}
    assert snapshot["toth"]["t"] == {"lower": 0.2, "upper": 3.0}
    assert detail["configuration"]["session_overrides"] == {
        "toth": {"t": {"lower": 0.2, "upper": 3.0}}
    }
    assert not any(parameter["customized"] for parameter in _model(client, "toth")["parameters"])
    assert detail["log"][-1].startswith("[INFO]")

    _run(client, {**base, "name": "Freundlich only", "models": ["freundlich"]})
    listing = client.get(
        "/api/v1/fitting/sessions", params={"page_size": 1, "sort": "name", "direction": "asc"}
    ).json()
    assert listing["total"] == 2
    assert [item["name"] for item in listing["items"]] == ["Freundlich only"]
    filtered = client.get(
        "/api/v1/fitting/sessions", params={"model": "toth", "query": "zeolite"}
    ).json()
    assert [item["id"] for item in filtered["items"]] == [detail["id"]]
    assert client.get(
        "/api/v1/fitting/sessions", params={"status": "failed"}
    ).json()["total"] == 0

    assert client.delete(f"/api/v1/fitting/sessions/{detail['id']}").status_code == 204
    assert client.get(f"/api/v1/fitting/sessions/{detail['id']}").status_code == 404
    assert client.get(f"/api/v1/datasets/{dataset_id}").status_code == 200
    assert client.get("/api/v1/fitting/sessions").json()["total"] == 1


###############################################################################
def test_sessions_left_running_are_failed_on_restart(client: TestClient) -> None:
    dataset_id = _import(client)
    service = client.app.state.core_container.fitting_service
    session_id = service.results.create_session(
        name="Interrupted",
        dataset_id=dataset_id,
        dataset_name="Two zeolites",
        isotherm_ids=[],
        models=["langmuir"],
        input_sha256="0" * 64,
        optimizer="trf",
        weighting="unweighted",
        max_evaluations=100,
        pressure_display_unit="bar",
        uptake_display_unit="mmol/g",
        configuration={},
    )
    assert client.delete(f"/api/v1/fitting/sessions/{session_id}").status_code == 409

    assert service.fail_interrupted_sessions() == 1
    detail = client.get(f"/api/v1/fitting/sessions/{session_id}").json()
    assert detail["status"] == "failed"
    assert detail["message"] == "Interrupted by application restart."
    assert detail["completed_at"] is not None
