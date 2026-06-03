# ADSMOD Runtime Modes

Last updated: 2026-06-03

## Supported Modes

### Local Web App Mode

- Backend: Uvicorn serving unified entrypoint `app.server.app:app`.
- Unified entrypoint composes core routes from `core_service` and training routes from `ml_service`.
- Frontends
  - Core UI in `app/client`, with dev default port `5173`.
  - ML UI in `app/ml_client`, with dev default port `5174`.
- Canonical launcher: `ADSMOD/start_on_windows.bat`.

### Core Service Mode

- Runs `core_service.app:app` without a frontend process.
- Intended for backend-only debugging or service integration work.

### ML Service Mode

- Runs `ml_service.app:app`.
- Exposes `/api/training/*` routes for dataset build and training management.

### Both Backend Services Mode

- Core service and ML service run together.
- This remains the target shape for future launcher coordination updates.

### Test Execution Mode

- Scripted runtime via `tests/run_tests.bat`.
- Starts backend or frontend only when not already running.

### Desktop Mode

- Tauri host process in `app/client/src-tauri/src/main.rs`.
- Spawns and monitors `app.server.app:app`, then loads the local backend URL in the webview.
- Backend serves packaged `app/client/dist` assets when `ADSMOD_TAURI_MODE=true`.

### Containerized Mode

- Not implemented in the current repository.
