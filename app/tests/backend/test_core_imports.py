from __future__ import annotations

import subprocess
import sys

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
