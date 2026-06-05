# ADSMOD Startup Procedures

Last updated: 2026-06-05

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
- syncs backend workspace dependencies into `app/server/.venv`
- installs frontend dependencies when needed
- exposes launch choices for core frontend + core service, ML frontend + ML service, or both stacks
- exposes setup and maintenance choices for core-only, ML-only, or shared operations

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
cd ADSMOD\app\ml_client
npm run dev
```

PowerShell:

```powershell
Set-Location ADSMOD/app/client
npm run dev
Set-Location ADSMOD/app/ml_client
npm run dev
```

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
