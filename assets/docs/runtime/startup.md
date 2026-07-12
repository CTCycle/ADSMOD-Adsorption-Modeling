# ADSMOD Startup Procedures

Last updated: 2026-07-11

## Recommended Local Web Startup

CMD:

```cmd
powershell -ExecutionPolicy Bypass -File ADSMOD\start_on_windows.ps1
```

PowerShell:

```powershell
& .\ADSMOD\start_on_windows.ps1
```

This menu-driven script:
- ensures portable runtimes under `runtimes/`
- syncs backend dependencies into `app/server/.venv`
- installs and builds frontend dependencies
- starts the unified backend and frontend preview
- exits after handing launch control to the local web stack instead of returning to the menu
- starts the frontend preview in the background and opens the browser after the UI responds
- respects `UI_PORT` overrides from `settings/.env` when launching the frontend dev server

## Setup And Maintenance

The same `start_on_windows.ps1` menu owns dependency installation, database initialization, tests, log removal, cache cleanup, and uninstall operations.

## API-Only Backend Startup

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
Set-Location ADSMOD/app/client
npm run dev
```

The frontend development server is an Angular CLI server with API proxy configuration loaded from `app/client/proxy.conf.cjs`.

## Test Startup

CMD:

```cmd
tests\run_tests.bat
```

PowerShell:

```powershell
.\tests\run_tests.bat
```
