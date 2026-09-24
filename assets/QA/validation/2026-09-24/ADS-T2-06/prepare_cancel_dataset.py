from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "app" / "tests"))

from e2e.test_datasets_api import _commit_sample  # noqa: E402


def main() -> None:
    config = json.loads(
        (REPO_ROOT / "app" / "resources" / "adsmod.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = config["runtime"]
    host = runtime["host"]
    port = runtime["backend_port"]
    backend_url = f"http://{host}:{port}"

    with tempfile.TemporaryDirectory(prefix="adsmod-fitting-cancel-") as temp_dir:
        fixture_path = Path(temp_dir) / "large-fitting-series.csv"
        with fixture_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(
                [
                    "experiment",
                    "adsorbent",
                    "adsorbate",
                    "temperature [K]",
                    "pressure [Pa]",
                    "uptake [mol/g]",
                ]
            )
            for index in range(200_000):
                pressure = 1.0 + index * 0.05
                uptake = 0.02 * 0.0004 * pressure / (1.0 + 0.0004 * pressure)
                uptake *= 1.0 + 0.03 * math.sin(index * 0.017)
                uptake *= 1.0 + 0.01 * math.cos(index * 0.0037)
                writer.writerow(
                    ["Cancellation Probe", "Carbon", "CO2", 298.15, pressure, uptake]
                )

        with sync_playwright() as playwright:
            api = playwright.request.new_context(base_url=backend_url)
            try:
                dataset = _commit_sample(
                    api,
                    str(fixture_path),
                    "validation_fitting_cancel_probe_200k",
                )
                experiments = api.get(
                    f"/api/v1/datasets/{dataset['id']}/experiments"
                )
                if not experiments.ok:
                    raise RuntimeError(experiments.text())
                eligible = next(
                    item
                    for item in experiments.json()["experiments"]
                    if item["fitting_eligible"]
                )
                print(
                    json.dumps(
                        {
                            "dataset_id": dataset["id"],
                            "dataset_name": dataset["name"],
                            "experiment_id": eligible["id"],
                            "experiment_name": eligible["name"],
                            "observation_count": eligible["observation_count"],
                        },
                        sort_keys=True,
                    )
                )
            finally:
                api.dispose()


if __name__ == "__main__":
    main()
