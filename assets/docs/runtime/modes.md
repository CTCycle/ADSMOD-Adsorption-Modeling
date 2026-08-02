# ADSMOD Runtime Modes

Last updated: 2026-08-02

## Supported Modes

### v3 Core Mode

- Uses the canonical `app/resources/adsmod.json` configuration.
- Runs the independent `adsmod-core` package and its health, capability, and internal snapshot endpoints.
- Does not require the ML package.

### v3 Core-ML Mode

- Uses the canonical v3 configuration with both `core` and `ml` services.
- ML consumes core-owned snapshots through the internal contract and must not access the core database directly.

The v3 modes are implemented package boundaries; launcher integration is still pending.

### Local Web App Mode

- The unified backend composition entrypoint is `app.server.app:app`.
- Frontend
  - Unified UI in `app/client`; the launcher uses the configured `5173` preview port.
  - `/api/training/*` proxy traffic targets the optional ML service.
- Canonical launcher: `start_on_windows.ps1` from the repository root.

### Core Service Mode

- Runs `core_service.app:app` without a frontend process.
- Intended for backend-only debugging or service integration work.
- Does not import `ml_service` or require ML-heavy dependencies.

### ML Service Mode

- Runs `ml_service.app:app`.
- Exposes `/api/training/*` routes for dataset build and training management.

### Both Backend Services Mode

- Core service and ML service run together.
- This remains the target shape for future launcher coordination updates.

### Test Execution Mode

- Scripted runtime via `app/tests/run_tests.bat`.
- Starts backend or frontend only when not already running.

### Containerized Mode

- Not implemented in the current repository.
