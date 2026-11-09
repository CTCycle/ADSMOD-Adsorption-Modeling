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
def test_running_app_loads_ml_runtime_only_on_first_training_request(
    tmp_path: Path,
) -> None:
    script = (
        "import sys; "
        "from pathlib import Path; "
        "from fastapi.testclient import TestClient; "
        "from server.app import create_app; "
        "from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings; "
        "heavy = lambda: any(n in sys.modules for n in ('torch', 'keras', 'sklearn', 'server.services.ml_container')); "
        f"config = AdsmodConfig.assemble(database=DatabaseConfig(embedded_database=True, sqlite_path='core.db'), storage_root=Path({str(tmp_path)!r}), settings=UserSettings()); "
        "client = TestClient(create_app(config)); client.__enter__(); "
        "assert client.get('/api/v1/system/capabilities').json()['features']['machine_learning'] is True; "
        "assert client.get('/api/v1/jobs').status_code == 200; "
        "assert not heavy(); "
        "assert client.get('/api/v1/training/status').status_code == 200; "
        "assert heavy(); "
        "client.__exit__(None, None, None)"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
