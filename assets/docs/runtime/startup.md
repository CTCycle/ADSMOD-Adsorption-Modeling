# ADSMOD Startup Procedures

Last updated: 2026-07-09

## Recommended Local Web Startup

CMD:

```cmd
ADSMOD\start_on_windows.bat
```

PowerShell:

```powershell
.\ADSMOD\start_on_windows.bat
```

This menu-driven script:
- ensures portable runtimes under `runtimes/`
- syncs scoped backend dependencies into `app/server/.venv`
- installs frontend dependencies when needed
- exposes launch choices for unified frontend + core service, unified frontend + ML service, or unified frontend + both services
- exits after handing launch control to the selected stack instead of returning to the menu
- starts frontend dev servers in the background and opens the browser after the selected UI responds
- respects `UI_PORT` overrides from `settings/.env` when launching the frontend dev server

## Setup And Maintenance

CMD:

```cmd
ADSMOD\setup_and_maintenance.bat
```

PowerShell:

```powershell
.\ADSMOD\setup_and_maintenance.bat
```

This separate menu-driven script owns setup, test, cleanup, and uninstall operations.

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

## Tauri Packaging Startup

CMD:

```cmd
release\tauri\build_with_tauri.bat
```

PowerShell:

```powershell
.\release\tauri\build_with_tauri.bat
```
