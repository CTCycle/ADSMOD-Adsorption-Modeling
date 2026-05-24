# ADSMOD Runtime Modes

Last updated: 2026-05-24

## 1. Supported Modes

### Local web app mode (primary)

- Backend: Uvicorn serving `core_service.app:app` (and `ml_service.app:app` for training APIs).
- Frontend: Vite preview (built `client/dist`) or Vite dev server.
- Canonical launcher: `ADSMOD/start_on_windows.bat`.

### Core service mode (API-only)

- Core backend via `core_service.app:app` (no frontend process required).
- Root behavior depends on host mode in backend startup logic.

### ML service mode (API-only)

- ML backend via `ml_service.app:app`.
- Exposes `/api/training/*` routes for dataset build and training management.

### Both backend services mode

- Core service and ML service run together.
- This is the target for future Stage 3 launcher menu updates.

### Test execution mode

- Scripted test runtime via `tests/run_tests.bat`.
- Starts backend/frontend only when not already running.

### Desktop mode (Tauri, Windows)

- Tauri host process in `client/src-tauri/src/main.rs`.
- Spawns and monitors backend process, then loads local backend URL in the webview.
- Packaging flow via `release/tauri/build_with_tauri.bat`.

### Containerized mode

- Not implemented in current repository (no Docker runtime configuration present).

## 2. Startup Procedures

### Local web app (recommended)

CMD:

```cmd
ADSMOD\start_on_windows.bat
```

PowerShell:

```powershell
.\ADSMOD\start_on_windows.bat
```

What it does:

- Ensures portable runtimes (`runtimes/python`, `runtimes/uv`, `runtimes/nodejs`).
- Syncs backend workspace deps with `uv` into `app/server/.venv`.
- Installs frontend deps, builds `client/dist` when missing.
- Starts backend and frontend on configured host/ports.

### API-only backend

CMD:

```cmd
app\server\.venv\Scripts\python.exe -m uvicorn core_service.app:app --host 127.0.0.1 --port 6045
```

PowerShell:

```powershell
.\app\server\.venv\Scripts\python.exe -m uvicorn core_service.app:app --host 127.0.0.1 --port 6045
```

### Frontend development server

CMD:

```cmd
cd ADSMOD\client
npm run dev
```

PowerShell:

```powershell
Set-Location ADSMOD/client
npm run dev
```

### Tests

CMD:

```cmd
tests\run_tests.bat
```

PowerShell:

```powershell
.\tests\run_tests.bat
```

### Tauri desktop packaging

CMD:

```cmd
release\tauri\build_with_tauri.bat
```

PowerShell:

```powershell
.\release\tauri\build_with_tauri.bat
```

## 3. Environment Variables and Configuration Requirements

Primary runtime env file: `ADSMOD/settings/.env`

Current keys used by launcher/runtime:

- `CORE_SERVICE_HOST`, `CORE_SERVICE_PORT`, `CORE_SERVICE_RELOAD`
- `ML_SERVICE_HOST`, `ML_SERVICE_PORT`, `ML_SERVICE_RELOAD`
- `FASTAPI_HOST`, `FASTAPI_PORT` (temporary compatibility keys)
- `UI_HOST`, `UI_PORT`
- `OPTIONAL_DEPENDENCIES`
- `MPLBACKEND`
- `KERAS_BACKEND`
- `VITE_API_BASE_URL`

Static application settings file: `ADSMOD/settings/configurations.json`

- Database mode and connection settings.
- Job polling interval.
- Dataset/NIST/fitting/training defaults.

## 4. Configuration Differences by Mode

- Local launcher mode:
  - Uses `.env` host/port values.
  - Core backend startup uses `CORE_SERVICE_*` values (with compatibility fallback to `FASTAPI_*`).
  - Runs backend and frontend as separate processes.
- API-only mode:
  - No frontend process required.
  - Best for backend debugging and service integration.
- Tauri mode:
  - Sets `ADSMOD_TAURI_MODE=true` for backend spawn path.
  - Serves packaged frontend assets and local backend inside desktop shell.
- Tests mode:
  - Reads `.env`, normalizes wildcard hosts (`0.0.0.0`, `::`) to `127.0.0.1` for client access.

## 5. Interoperability

- Frontend calls backend using `/api` path.
- Vite `server.proxy` and `preview.proxy` forward `/api` to backend host/port from env settings.
- Tauri runtime waits for backend port readiness, then redirects window to backend root URL.
- Shared storage/services across modes:
  - Database (`resources/database.db` for embedded mode).
  - Checkpoints (`resources/checkpoints`).
  - Runtime env/config files (`settings/.env`, `settings/configurations.json`).

## 6. Limitations and Constraints

- Windows-first script/tooling flow (`.bat`, PowerShell helpers).
- First launch can be slow due to dependency/runtime provisioning.
- Desktop packaging requires working Rust/Cargo toolchain.
- No container runtime target currently implemented.

## 7. Deployment and Packaging Notes

- Desktop packaging builds and exports Windows artifacts under `release/windows`.
- Tauri bundle includes staged runtime resources (server code, scripts, settings, dist assets, runtime binaries).
- Backend workspace lock state is maintained in `app/server/uv.lock`.

