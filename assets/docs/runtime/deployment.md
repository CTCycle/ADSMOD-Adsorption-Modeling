# ADSMOD Local Deployment

Last updated: 2026-07-11

## Interoperability

- Unified frontend calls non-training backend routes through `/api`.
- Unified frontend routes `/api/training/*` to the optional ML service.
- The local launcher waits for backend health and frontend readiness before opening the browser.

## Shared Runtime Resources

- Database: `resources/database.db` for embedded mode
- Checkpoints: `resources/checkpoints`
- Runtime env and config files: `settings/.env`, `settings/core_service.json`, and `settings/ml_service.json`

## Local Deployment Notes

- The supported end-user runtime is the Windows local web launcher at `start_on_windows.ps1`.
- Portable Python, uv, and Node.js are provisioned under `runtimes/`.
- The frontend is built before launch and served by the hidden preview process.
- Backend visibility is controlled by `BACKEND_VISIBLE` in `settings/.env`.
- Backend dependency state is locked in `app/server/uv.lock`.

## Constraints

- The repository is Windows-first and uses a PowerShell launcher plus the batch test runner.
- First launch can be slow because runtime binaries and dependencies may need provisioning.
- No container runtime target is currently implemented.
