from __future__ import annotations

import os
import subprocess
import sys

###############################################################################
def test_core_service_does_not_import_ml_libs() -> None:
    before = set(sys.modules)
    __import__("core_service.app")
    loaded = set(sys.modules) - before
    assert "torch" not in loaded
    assert "keras" not in loaded
    assert "sklearn" not in loaded


def test_core_only_apps_do_not_load_ml_packages() -> None:
    environment = os.environ.copy()
    environment.pop("ADSMOD_ENABLE_ML", None)
    script = (
        "import sys; "
        "import core_service.app; "
        "import app.server.app; "
        "assert not any(name == 'ml_service' or name.startswith('ml_service.') "
        "for name in sys.modules); "
        "assert not any(name in sys.modules for name in ('torch', 'keras', 'sklearn'))"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
