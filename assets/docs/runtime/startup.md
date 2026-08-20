# ADSMOD Startup Procedures

Last updated: 2026-08-20

## Recommended Local Web Startup

CMD:

```cmd
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1
```

PowerShell:

```powershell
& .\start_on_windows.ps1
```

This menu-driven script:
- ensures portable runtimes under `runtimes/`
- syncs backend dependencies into `app/server/.venv`
- installs frontend dependencies as needed
- repairs missing or unusable dependencies and frontend build output during launch
- always rebuilds the frontend when menu option 2, **Install / update dependencies**, is executed
- provides menu option 3, **Rebuild frontend**, to install frontend packages and rebuild the bundle without syncing backend dependencies
- starts the unified backend and frontend preview
- exits after handing launch control to the local web stack instead of returning to the menu
- starts the frontend preview in the background and opens the browser after the UI responds
- uses `runtime.core_port`, `runtime.ml_port`, and `runtime.frontend_port` from
  `$ADSMOD_RESOURCES_DIR/adsmod.json` (`app/resources` by default)

Before launch, the script creates `settings/.env` from
`settings/.env.example` only when the file is missing. Existing environment
files are preserved.

Set `ADSMOD_RESOURCES_DIR` in `settings/.env` when the canonical resource
directory, including the embedded SQLite database, should live elsewhere. A
relative value is resolved from the repository root, and the selected
directory must contain `adsmod.json`.

## Database Startup Rules

- Startup runs the synchronous Alembic coordinator before serving requests.
- SQLite creates missing storage and upgrades it to the packaged head. An
  existing pre-Alembic file is adopted only after an exact structural check;
  mismatches fail without stamping or changing the schema.
- PostgreSQL startup creates the configured database when the configured role
  permits it, then applies pending migrations. Advisory locks serialize
  creation and upgrades across application instances.
- The menu's **Initialize database** command invokes the same coordinator and
  is safe to repeat for fresh, legacy, outdated, or current databases.
- Migration failures abort startup and leave transactional changes rolled back.

## Setup And Maintenance

The same `start_on_windows.ps1` menu owns dependency installation, database initialization, tests, log removal, cache cleanup, and uninstall operations.

Disposable cache locations are fixed by the launcher and test runner:

- `runtimes/cache` contains uv, npm, pip, and runtime temporary files.
- `app/tests/cache` contains Angular, pytest, Ruff, mypy, Python bytecode,
  coverage, and pytest temporary files.

The **Clear cache** menu option removes the contents of both locations and
cleans up legacy Python cache directories. Files that require administrator
rights or are otherwise locked are warned about and skipped so cleanup can
continue.

## Unified Backend Startup

From the repository root:

CMD:

```cmd
app\server\.venv\Scripts\python.exe -m uvicorn app.server.app:app --host 127.0.0.1 --port 6045
```

PowerShell:

```powershell
.\app\server\.venv\Scripts\python.exe -m uvicorn app.server.app:app --host 127.0.0.1 --port 6045
```

Set `ADSMOD_ENABLE_ML=true` before starting the unified backend when the ML
routes should be mounted in the same process. Otherwise, the core-only runtime
keeps training unavailable while datasets and fitting remain usable.

## Core Service Startup

CMD:

```cmd
app\server\.venv\Scripts\python.exe -m uvicorn core_service.app:app --host 127.0.0.1 --port 6045
```

PowerShell:

```powershell
.\app\server\.venv\Scripts\python.exe -m uvicorn core_service.app:app --host 127.0.0.1 --port 6045
```

## Frontend Development Servers

CMD:

```cmd
cd ADSMOD\app\client
npm run dev
```

PowerShell:

```powershell
Set-Location app/client
npm run dev
```

The frontend development server is an Angular CLI server with API proxy configuration loaded from `app/client/proxy.conf.cjs`.

## Test Startup

CMD:

```cmd
app\tests\run_tests.bat
```

PowerShell:

```powershell
.\app\tests\run_tests.bat
```
