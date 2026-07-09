# ADSMOD Deployment And Packaging

Last updated: 2026-07-09

## Interoperability

- Unified frontend calls non-training backend routes through `/api`.
- Unified frontend routes `/api/training/*` to the optional ML service.
- Tauri waits for backend port readiness, then redirects the window to the backend root URL.

## Shared Runtime Resources

- Database: `resources/database.db` for embedded mode
- Checkpoints: `resources/checkpoints`
- Runtime env and config files: `settings/.env`, `settings/core_service.json`, and `settings/ml_service.json`

## Packaging Notes

- Desktop packaging flows through `release/tauri/build_with_tauri.bat`.
- Windows packaging outputs artifacts under `release/windows`.
- The Tauri bundle stages server code, scripts, settings, Angular frontend dist assets, and runtime binaries.
- ML registration in the unified packaged backend is conditional on `ADSMOD_ENABLE_ML=true`.
- Backend dependency state is locked in `app/server/uv.lock`.

## Constraints

- The repository is Windows-first and relies on `.bat` and PowerShell workflows.
- First launch can be slow because runtime binaries and dependencies may need provisioning.
- Desktop packaging requires a working Rust and Cargo toolchain.
- No container runtime target is currently implemented.
