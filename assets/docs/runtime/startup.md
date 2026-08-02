# ADSMOD Startup Procedures

Last updated: 2026-08-02

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
- installs and builds frontend dependencies
- starts the unified backend and frontend preview
- exits after handing launch control to the local web stack instead of returning to the menu
- starts the frontend preview in the background and opens the browser after the UI responds
- uses `runtime.core_port`, `runtime.ml_port`, and `runtime.frontend_port` from `app/resources/adsmod.json`

## Setup And Maintenance

The same `start_on_windows.ps1` menu owns dependency installation, database initialization, tests, log removal, cache cleanup, and uninstall operations.

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
