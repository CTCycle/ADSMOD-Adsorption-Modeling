# ADSMOD Startup Procedures

Last updated: 2026-06-03

## Recommended Local Web Startup

CMD:

```cmd
ADSMOD\start_on_windows.bat
```

PowerShell:

```powershell
.\ADSMOD\start_on_windows.bat
```

This launcher:

- ensures portable runtimes under `runtimes/`
- syncs backend workspace dependencies into `app/server/.venv`
- installs frontend dependencies when needed
- builds frontend assets when required
- starts the unified backend and frontend processes

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
