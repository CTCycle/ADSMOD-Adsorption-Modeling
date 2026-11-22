# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


###############################################################################
def test_core_app_import_does_not_load_ml_runtime() -> None:
    script = (
        "import sys; "
        "import server.app; "
        "assert 'server.services.ml_container' not in sys.modules; "
        "assert 'server.api.training' not in sys.modules; "
        "assert not any(name in sys.modules for name in ('torch', 'keras', 'sklearn'))"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


###############################################################################
def test_training_inspection_endpoints_do_not_load_ml_runtime(
    tmp_path: Path,
) -> None:
    script = (
        "import sys; "
        "from pathlib import Path; "
        "from fastapi.testclient import TestClient; "
        "from server.app import create_app; "
        "from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings; "
        "heavy = lambda: any(n in sys.modules for n in ('torch', 'keras', 'sklearn')); "
        f"config = AdsmodConfig.assemble(database=DatabaseConfig(embedded_database=True, sqlite_path='core.db'), storage_root=Path({str(tmp_path)!r}), settings=UserSettings()); "
        "import time; "
        "client = TestClient(create_app(config)); client.__enter__(); "
        "status = lambda: client.get('/api/v1/deep-learning/status').json()['status']; "
        "deadline = time.monotonic() + 180; "
        "[time.sleep(0.5) for _ in iter(lambda: status() == 'verifying' and time.monotonic() < deadline, False)]; "
        "assert client.get('/api/v1/system/capabilities').json()['features']['machine_learning'] is True; "
        "assert client.get('/api/v1/jobs').status_code == 200; "
        "assert not heavy(); "
        "assert 'server.services.ml_container' not in sys.modules; "
        "paths = ('configuration', 'status', 'checkpoints', 'datasets', 'processed-datasets', 'dataset-sources', 'jobs'); "
        "codes = {p: client.get(f'/api/v1/training/{p}').status_code for p in paths}; "
        "assert set(codes.values()) == {200}, codes; "
        "assert 'server.services.ml_container' in sys.modules; "
        "assert not heavy(), [n for n in ('torch', 'keras', 'sklearn') if n in sys.modules]; "
        "client.__exit__(None, None, None)"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
