"""Measure bounded local public-data, import, and fitting-polling paths."""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from fastapi.testclient import TestClient

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY_ROOT / "app"))

from server.app import create_app  # noqa: E402
from server.configurations.settings import StorageConfig, load_config  # noqa: E402


def timed_request(
    request: Callable[[], Any],
) -> tuple[Any, float]:
    started = time.perf_counter()
    response = request()
    return response, round((time.perf_counter() - started) * 1000, 3)


def main() -> int:
    config = load_config(REPOSITORY_ROOT / "resources/adsmod.json")
    fixture = (REPOSITORY_ROOT / "app/tests/fixtures/sample_adsorption.csv").read_bytes()
    dataset_name = f"t5_04_{uuid.uuid4().hex[:10]}"

    with TemporaryDirectory(prefix="adsmod-t5-04-") as storage_dir:
        isolated_config = config.model_copy(
            update={"storage": StorageConfig(root=Path(storage_dir))}
        )
        with TestClient(create_app(isolated_config)) as client:
            ready, ready_ms = timed_request(lambda: client.get("/health/ready"))
            assert ready.status_code == 200, ready.text

            preview, preview_ms = timed_request(
                lambda: client.post(
                    "/api/v1/datasets/import/preview",
                    files={
                        "file": (
                            f"{dataset_name}.csv",
                            fixture,
                            "text/csv",
                        )
                    },
                )
            )
            assert preview.status_code == 200, preview.text
            preview_payload = preview.json()
            column_roles = {
                column["name"]: column["proposed_role"]
                for column in preview_payload["columns"]
            }
            mapping = {
                "dataset_name": dataset_name,
                "structure": preview_payload["detected_structure"],
                "column_roles": column_roles,
                "grouping_columns": preview_payload.get("proposed_grouping_columns")
                or [
                    name
                    for name, role in column_roles.items()
                    if role in {"experiment_id", "experiment_name"}
                ],
                "pressure_basis": preview_payload.get("proposed_pressure_basis")
                or "absolute",
                "unit_overrides": {
                    column["proposed_role"]: column["detected_unit"]
                    for column in preview_payload["columns"]
                    if column.get("detected_unit")
                    and column["proposed_role"]
                    in {"pressure", "uptake", "temperature"}
                },
                "decimal_separator": ".",
                "duplicate_policy": "keep",
            }
            validation, validation_ms = timed_request(
                lambda: client.post(
                    "/api/v1/datasets/import/validate",
                    data={"mapping": json.dumps(mapping)},
                    files={
                        "file": (
                            f"{dataset_name}.csv",
                            fixture,
                            "text/csv",
                        ),
                    },
                )
            )
            assert validation.status_code == 200, validation.text
            assert validation.json()["status"] == "valid"

            committed, commit_ms = timed_request(
                lambda: client.post(
                    "/api/v1/datasets/import/commit",
                    data={"mapping": json.dumps(mapping)},
                    files={
                        "file": (
                            f"{dataset_name}.csv",
                            fixture,
                            "text/csv",
                        ),
                    },
                )
            )
            assert 200 <= committed.status_code < 300, committed.text
            dataset = committed.json()["dataset"]

            public_data: dict[str, Any] = {}
            for name, path in {
                "sources": "/api/v1/public-data/sources?check_health=false",
                "adsorption": "/api/v1/public-data/adsorption?page=1&page_size=25",
                "materials": "/api/v1/public-data/materials?page=1&page_size=25",
                "chemicals": "/api/v1/public-data/chemicals?page=1&page_size=25",
                "structures": "/api/v1/public-data/structures?page=1&page_size=25",
            }.items():
                response, elapsed_ms = timed_request(lambda path=path: client.get(path))
                assert response.status_code == 200, response.text
                payload = response.json()
                public_data[name] = {
                    "elapsed_ms": elapsed_ms,
                    "status": response.status_code,
                    "item_count": len(
                        payload.get("items", payload.get("sources", []))
                    ),
                    "total": payload.get("pagination", {}).get("total"),
                }

            experiments, experiments_ms = timed_request(
                lambda: client.get(
                    f"/api/v1/datasets/{dataset['id']}/experiments"
                )
            )
            assert experiments.status_code == 200, experiments.text
            experiment = next(
                item
                for item in experiments.json()["experiments"]
                if item["fitting_eligible"]
            )
            fitting_started, fitting_start_ms = timed_request(
                lambda: client.post(
                    "/api/v1/fitting/run",
                    json={
                        "dataset_id": dataset["id"],
                        "isotherm_id": experiment["id"],
                        "models": ["langmuir"],
                        "optimizer": "trf",
                        "max_evaluations": 50,
                    },
                )
            )
            assert fitting_started.status_code == 200, fitting_started.text
            job_id = fitting_started.json()["job_id"]
            poll_count = 0
            poll_started = time.perf_counter()
            terminal_status: dict[str, Any] | None = None
            while time.perf_counter() - poll_started < 30:
                status_response = client.get(f"/api/v1/fitting/jobs/{job_id}")
                assert status_response.status_code == 200, status_response.text
                poll_count += 1
                status_payload = status_response.json()
                if status_payload.get("status") in {
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    terminal_status = status_payload
                    break
                time.sleep(0.05)
            assert terminal_status is not None
            assert terminal_status["status"] == "completed", terminal_status
            polling_ms = round((time.perf_counter() - poll_started) * 1000, 3)

            deleted, delete_ms = timed_request(
                lambda: client.delete(f"/api/v1/datasets/{dataset['id']}")
            )
            assert deleted.status_code == 204, deleted.text

            print(
                json.dumps(
                    {
                        "fixture": "app/tests/fixtures/sample_adsorption.csv",
                        "storage": "TemporaryDirectory",
                        "health_ready_ms": ready_ms,
                        "import": {
                            "preview_ms": preview_ms,
                            "validation_ms": validation_ms,
                            "commit_ms": commit_ms,
                            "experiment_count": dataset["experiment_count"],
                            "observation_count": dataset["observation_count"],
                        },
                        "public_data": public_data,
                        "polling": {
                            "start_ms": fitting_start_ms,
                            "poll_count": poll_count,
                            "elapsed_ms": polling_ms,
                            "server_poll_interval_seconds": fitting_started.json().get(
                                "poll_interval"
                            ),
                            "terminal_status": terminal_status["status"],
                        },
                        "experiment_lookup_ms": experiments_ms,
                        "cleanup_delete_ms": delete_ms,
                    },
                    sort_keys=True,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
