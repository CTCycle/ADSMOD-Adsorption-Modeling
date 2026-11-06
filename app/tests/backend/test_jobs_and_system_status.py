# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from server.api.jobs import acquisition_job
from server.app import create_app
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings


###############################################################################
def _config(tmp_path: Path) -> AdsmodConfig:
    return AdsmodConfig.assemble(
        database=DatabaseConfig(embedded_database=True, sqlite_path="data/test.db"),
        storage_root=tmp_path,
        settings=UserSettings(),
    )


###############################################################################
def test_system_status_reports_process_metrics_and_database_size(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        payload = client.get("/api/v1/system/status").json()
    assert payload["memory_mb"] > 0
    assert 0 <= payload["cpu_percent"] <= 100
    assert payload["database_size_bytes"] > 0


###############################################################################
def test_jobs_endpoint_lists_managed_jobs_with_wall_clock_times(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        manager = client.app.state.core_container.job_manager
        job_id = manager.start_job("fitting", lambda: {"fitted": True})
        deadline = time.monotonic() + 5
        while manager.is_job_running("fitting") and time.monotonic() < deadline:
            time.sleep(0.02)
        payload = client.get("/api/v1/jobs").json()
    job = next(item for item in payload["jobs"] if item["id"] == job_id)
    assert job["kind"] == "fitting"
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["finished_at"] is not None
    assert payload["active_count"] == 0


###############################################################################
def test_acquisition_job_maps_sync_run_counts_and_details() -> None:
    now = datetime.now(UTC)
    job = acquisition_job(
        {
            "id": "run-1",
            "source_key": "nist",
            "operation": "nist_category",
            "parameters": {"category": "materials"},
            "status": "partially_completed",
            "progress": 100.0,
            "processed_count": 10,
            "imported_count": 7,
            "skipped_count": 2,
            "failed_count": 1,
            "checkpoint": {},
            "cancel_requested": False,
            "error_detail": None,
            "created_at": now.replace(tzinfo=None),
            "started_at": now,
            "finished_at": now,
        }
    )
    assert (job.kind, job.source, job.status) == (
        "acquisition",
        "nist",
        "partially_completed",
    )
    assert (job.saved_count, job.skipped_count, job.failed_count) == (7, 2, 1)
    assert job.created_at.tzinfo is not None
    assert job.run is not None and job.run.id == "run-1"
