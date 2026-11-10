# ADSMOD local deployment

Last updated: 2026-11-10

The supported deployment is the Windows local web launcher. Use start_on_windows.bat to launch; it invokes setup_and_maintenance.ps1 -Launch through PowerShell 7. Run setup_and_maintenance.ps1 directly for maintenance actions:

```powershell
.\start_on_windows.bat
```

The launcher provisions portable Python, uv, and Node.js under `runtimes/`,
synchronizes `app/server` into `app/server/.venv`, builds the Angular client,
starts one FastAPI backend that also serves the production frontend on the
backend port, and supervises that process until Ctrl+C.

The launcher always installs the core dependencies only. Optional deep learning
packages (PyTorch, Keras, scikit-learn) are installed, repaired and removed
from Settings, into the same virtual environment, by a detached installer that
runs while ADSMOD is closed and then restarts it. Windows x64 and Python 3.14
only. The same backend entry point runs either way; no second server exists.

The browser uses same-origin `/api/v1` requests. The unified backend exposes
`/health/live`, `/health/ready`, and `/api/v1/system/capabilities`. The frontend
uses the capabilities response to decide whether ML functionality is
available.

Embedded SQLite is stored below the configured storage root. PostgreSQL remains
available through the typed database configuration; startup applies the
packaged Alembic history and does not infer an unknown schema.

All disposable uv, npm, Python, pytest, frontend, browser, coverage, and
temporary caches belong under the single repository-local hierarchy
`runtimes/cache`. Persistent application data remains below the configured
storage root. No container deployment target is currently implemented.
