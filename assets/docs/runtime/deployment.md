# ADSMOD Local Deployment

Last updated: 2026-07-20

## Interoperability

- Unified frontend calls non-training backend routes through `/api`.
- Unified frontend routes `/api/training/*` to the optional ML service.
- The local launcher waits for backend health and frontend readiness before opening the browser.

## Shared Runtime Resources

- Database: `resources/database.db` for embedded mode
- Checkpoints: `resources/checkpoints`
- Runtime env and config files: `settings/.env`, `settings/core_service.json`, and `settings/ml_service.json`
- Canonical v3 configuration: `settings/adsmod.json` with `settings/adsmod.schema.json`

## Local Deployment Notes

- The supported end-user runtime is the Windows local web launcher at `start_on_windows.ps1`.
- Portable Python, uv, and Node.js are provisioned under `runtimes/`.
- The frontend is built before launch and served by the hidden preview process.
- Backend log visibility is controlled by `BACKEND_LOGS_VISIBLE` in `settings/.env` and defaults to `true` when absent.
- Backend dependency state is locked in `app/server/uv.lock`.
- v3 package dependencies are declared independently in `app/backend/common/pyproject.toml`, `app/backend/core/pyproject.toml`, and `app/backend/ml/pyproject.toml`.

## Constraints

- The repository is Windows-first and uses a PowerShell launcher plus the batch test runner.
- First launch can be slow because runtime binaries and dependencies may need provisioning.
- No container runtime target is currently implemented.
- The Windows launcher currently starts the transitional `app/server` runtime; v3 package launcher integration is not yet implemented.
