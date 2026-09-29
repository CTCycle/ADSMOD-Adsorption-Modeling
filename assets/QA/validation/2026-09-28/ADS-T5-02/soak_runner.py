"""Bounded serial dataset/fitting soak used by the 2026-09-28 QA campaign."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from playwright.sync_api import APIRequestContext, sync_playwright

BASE_URL = os.environ.get("ADSMOD_SOAK_BACKEND_URL", "http://127.0.0.1:6045")
FIXTURE = Path(os.environ.get("ADSMOD_SOAK_FIXTURE", "app/tests/fixtures/sample_adsorption.csv"))
OUTPUT = Path(os.environ.get("ADSMOD_SOAK_OUTPUT", "assets/QA/validation/2026-09-28/ADS-T5-02/soak-results.json"))
SOAK_SECONDS = float(os.environ.get("ADSMOD_SOAK_SECONDS", "600"))
CYCLE_INTERVAL_SECONDS = float(os.environ.get("ADSMOD_SOAK_CYCLE_INTERVAL", "12"))
FITTING_MODELS = [
    "langmuir", "sips", "freundlich", "temkin", "toth",
    "dubinin_radushkevich", "dual_site_langmuir", "redlich_peterson", "jovanovic",
]


###############################################################################
def payload(response: Any) -> dict[str, Any]:
    try:
        value = response.json()
    except Exception:  # noqa: BLE001
        value = {"text": response.text()[:300]}
    return value if isinstance(value, dict) else {"value": value}


###############################################################################
def build_mapping(api: APIRequestContext, content: bytes, name: str) -> dict[str, Any]:
    response = api.post(
        "/api/v1/datasets/import/preview",
        multipart={"file": {"name": f"{name}.csv", "mimeType": "text/csv", "buffer": content}},
    )
    if not response.ok:
        raise RuntimeError(f"preview status={response.status} payload={payload(response)}")
    preview = payload(response)
    roles = {column["name"]: column["proposed_role"] for column in preview["columns"]}
    return {
        "dataset_name": name,
        "structure": preview["detected_structure"],
        "column_roles": roles,
        "grouping_columns": preview.get("proposed_grouping_columns") or [
            column_name for column_name, role in roles.items()
            if role in {"experiment_id", "experiment_name"}
        ],
        "pressure_basis": preview.get("proposed_pressure_basis") or "absolute",
        "unit_overrides": {
            column["proposed_role"]: column["detected_unit"]
            for column in preview["columns"]
            if column.get("detected_unit")
            and column["proposed_role"] in {"pressure", "uptake", "temperature"}
        },
        "decimal_separator": ".",
        "duplicate_policy": "keep",
    }


###############################################################################
def commit_dataset(api: APIRequestContext, content: bytes, name: str) -> tuple[dict[str, Any], dict[str, Any], float]:
    started = time.monotonic()
    mapping = build_mapping(api, content, name)
    fields = {
        "mapping": json.dumps(mapping),
        "file": {"name": f"{name}.csv", "mimeType": "text/csv", "buffer": content},
    }
    validation = api.post("/api/v1/datasets/import/validate", multipart=fields)
    if not validation.ok or payload(validation).get("status") != "valid":
        raise RuntimeError(f"validate status={validation.status} payload={payload(validation)}")
    committed = api.post("/api/v1/datasets/import/commit", multipart=fields)
    if not committed.ok:
        raise RuntimeError(f"commit status={committed.status} payload={payload(committed)}")
    return payload(committed)["dataset"], mapping, time.monotonic() - started


###############################################################################
def wait_ready(api: APIRequestContext, timeout_seconds: float = 45.0) -> float:
    started = time.monotonic()
    deadline = started + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = api.get("/health/ready", timeout=2000)
            if response.ok and payload(response).get("state") == "ready":
                return time.monotonic() - started
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    raise TimeoutError("backend did not become ready within the bounded restart window")


###############################################################################
def poll_job(api: APIRequestContext, job_id: str, timeout_seconds: float = 45.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = api.get(f"/api/v1/fitting/jobs/{job_id}")
        if not response.ok:
            raise RuntimeError(f"job status={response.status} payload={payload(response)}")
        status = payload(response)
        if status.get("status") in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} did not reach a terminal state")


###############################################################################
def start_fit(api: APIRequestContext, dataset_id: int, isotherm_id: int) -> tuple[str, int]:
    request = {
        "dataset_id": dataset_id,
        "isotherm_id": isotherm_id,
        "models": FITTING_MODELS,
        "optimizer": "trf",
        "max_evaluations": 1000,
    }
    response = api.post("/api/v1/fitting/run", data=request)
    if not response.ok:
        raise RuntimeError(f"fit start status={response.status} payload={payload(response)}")
    job_id = str(payload(response)["job_id"])
    duplicate = api.post("/api/v1/fitting/run", data=request)
    if duplicate.status != 400:
        raise RuntimeError(f"duplicate fit status={duplicate.status} payload={payload(duplicate)}")
    return job_id, duplicate.status


###############################################################################
def main() -> None:
    content = FIXTURE.read_bytes()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    deadline = time.monotonic() + SOAK_SECONDS
    result: dict[str, Any] = {
        "started_at_epoch": started_at,
        "duration_target_seconds": SOAK_SECONDS,
        "cycle_interval_seconds": CYCLE_INTERVAL_SECONDS,
        "backend_url": BASE_URL,
        "fixture": str(FIXTURE),
        "counts": Counter(),
        "http_outcomes": Counter(),
        "events": [],
        "failures": [],
        "readiness_recoveries": [],
        "sentinel": {},
        "final": {},
    }

    with sync_playwright() as playwright:
        api = playwright.request.new_context(base_url=BASE_URL)
        try:
            ready_seconds = wait_ready(api)
            result["events"].append({"event": "initial_ready", "seconds": round(ready_seconds, 3)})
            for dataset in payload(api.get("/api/v1/datasets")).get("datasets", []):
                if str(dataset.get("name", "")).startswith("t5_residual_"):
                    deleted = api.delete(f"/api/v1/datasets/{dataset['id']}")
                    result["http_outcomes"][f"delete_preexisting_{deleted.status}"] += 1

            sentinel_name = f"t5_residual_sentinel_{uuid.uuid4().hex[:8]}"
            sentinel, _, commit_seconds = commit_dataset(api, content, sentinel_name)
            result["sentinel"] = {
                "dataset_id": sentinel["id"], "name": sentinel["name"],
                "experiment_count": sentinel["experiment_count"],
                "observation_count": sentinel["observation_count"],
                "commit_seconds": round(commit_seconds, 3),
            }
            result["counts"]["sentinel_commits"] += 1
            experiments = payload(api.get(f"/api/v1/datasets/{sentinel['id']}/experiments"))["experiments"]
            target = next(item for item in experiments if item["fitting_eligible"])
            sentinel_job, sentinel_duplicate = start_fit(api, sentinel["id"], target["id"])
            sentinel_status = poll_job(api, sentinel_job)
            result["sentinel"].update({
                "isotherm_id": target["id"], "job_id": sentinel_job,
                "job_status": sentinel_status["status"],
                "duplicate_job_status": sentinel_duplicate,
            })
            result["counts"]["sentinel_fitting_runs"] += 1

            while time.monotonic() < deadline:
                cycle_started = time.monotonic()
                cycle: dict[str, Any] = {
                    "name": f"t5_residual_cycle_{uuid.uuid4().hex[:8]}",
                    "started_at_epoch": time.time(),
                }
                try:
                    dataset, mapping, commit_seconds = commit_dataset(api, content, cycle["name"])
                    cycle.update({
                        "dataset_id": dataset["id"],
                        "commit_seconds": round(commit_seconds, 3),
                        "experiment_count": dataset["experiment_count"],
                        "observation_count": dataset["observation_count"],
                    })
                    result["counts"]["dataset_commits"] += 1
                    duplicate = api.post(
                        "/api/v1/datasets/import/commit",
                        multipart={
                            "mapping": json.dumps(mapping),
                            "file": {"name": f"{cycle['name']}.csv", "mimeType": "text/csv", "buffer": content},
                        },
                    )
                    result["http_outcomes"][f"duplicate_dataset_{duplicate.status}"] += 1
                    if duplicate.status != 400:
                        raise RuntimeError(f"duplicate dataset status={duplicate.status}")
                    result["counts"]["duplicate_dataset_checks"] += 1
                    listed = payload(api.get("/api/v1/datasets"))["datasets"]
                    matches = [item for item in listed if item["name"] == cycle["name"]]
                    if len(matches) != 1 or matches[0]["id"] != dataset["id"]:
                        raise RuntimeError(f"dataset list mismatch for {cycle['name']}")
                    result["counts"]["dataset_list_checks"] += 1
                    experiments = payload(api.get(f"/api/v1/datasets/{dataset['id']}/experiments"))["experiments"]
                    target = next(item for item in experiments if item["fitting_eligible"])
                    job_id, duplicate_fit_status = start_fit(api, dataset["id"], target["id"])
                    status = poll_job(api, job_id)
                    cycle.update({
                        "isotherm_id": target["id"], "job_id": job_id,
                        "job_status": status["status"], "job_terminal_error": status.get("error"),
                        "duplicate_fit_status": duplicate_fit_status,
                    })
                    result["counts"]["fitting_runs"] += 1
                    result["counts"]["duplicate_fit_checks"] += 1
                    if status["status"] != "completed":
                        raise RuntimeError(f"job terminal status={status['status']}")
                    deleted = api.delete(f"/api/v1/datasets/{dataset['id']}")
                    result["http_outcomes"][f"dataset_delete_{deleted.status}"] += 1
                    if deleted.status != 204:
                        raise RuntimeError(f"dataset delete status={deleted.status}")
                    remaining = payload(api.get("/api/v1/datasets"))["datasets"]
                    if any(item["id"] == dataset["id"] for item in remaining):
                        raise RuntimeError(f"deleted dataset {dataset['id']} remained listed")
                    result["counts"]["delete_list_checks"] += 1
                    cycle["elapsed_seconds"] = round(time.monotonic() - cycle_started, 3)
                    result["events"].append(cycle)
                except Exception as exc:  # noqa: BLE001
                    cycle["elapsed_seconds"] = round(time.monotonic() - cycle_started, 3)
                    cycle["error"] = str(exc)
                    result["failures"].append(cycle)
                    result["events"].append(cycle)
                remaining_sleep = CYCLE_INTERVAL_SECONDS - (time.monotonic() - cycle_started)
                if remaining_sleep > 0:
                    time.sleep(min(remaining_sleep, max(0.0, deadline - time.monotonic())))
                try:
                    recovery_seconds = wait_ready(api, timeout_seconds=2.0)
                    if recovery_seconds > 0.8:
                        persisted = payload(api.get("/api/v1/datasets"))["datasets"]
                        result["readiness_recoveries"].append({
                            "seconds": round(recovery_seconds, 3),
                            "sentinel_present": any(item["id"] == sentinel["id"] for item in persisted),
                        })
                except TimeoutError:
                    pass

            final_experiments = payload(api.get(f"/api/v1/datasets/{sentinel['id']}/experiments"))["experiments"]
            final_jobs = payload(api.get("/api/v1/fitting/jobs"))
            result["final"] = {
                "sentinel_experiment_count": len(final_experiments),
                "active_jobs": [
                    job for job in final_jobs.get("jobs", [])
                    if job.get("status") not in {"completed", "failed", "cancelled"}
                ],
                "job_list_count": len(final_jobs.get("jobs", [])),
            }
            deleted = api.delete(f"/api/v1/datasets/{sentinel['id']}")
            result["http_outcomes"][f"sentinel_delete_{deleted.status}"] += 1
            result["final"]["sentinel_delete_status"] = deleted.status
            result["final"]["remaining_dataset_count"] = len(payload(api.get("/api/v1/datasets")).get("datasets", []))
        finally:
            api.dispose()

    result["finished_at_epoch"] = time.time()
    result["duration_seconds"] = round(result["finished_at_epoch"] - started_at, 3)
    result["counts"] = dict(result["counts"])
    result["http_outcomes"] = dict(result["http_outcomes"])
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
