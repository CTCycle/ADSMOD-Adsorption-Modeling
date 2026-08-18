# ADSMOD Local Deployment

Last updated: 2026-08-18

## Interoperability

- Unified frontend calls non-training backend routes through `/api`.
- Unified frontend routes `/api/training/*` to the optional ML service.
- The local launcher waits for backend health and frontend readiness before opening the browser.

## Shared Runtime Resources

- Resource directory: `app/resources` by default; override it with
  `ADSMOD_RESOURCES_DIR` in `settings/.env`
- Database: `<resource directory>/database.db` for embedded mode; PostgreSQL is
  initialized explicitly through the launcher
- Checkpoints: `<resource directory>/checkpoints`
- Operational environment template: `settings/.env.example`
- Canonical v3 configuration: `app/resources/adsmod.json` with `app/resources/adsmod.schema.json`

## Local Deployment Notes

- The supported end-user runtime is the Windows local web launcher at `start_on_windows.ps1`.
- Portable Python, uv, and Node.js are provisioned under `runtimes/`.
- The frontend is built before launch and served by the hidden Angular production-preview process.
- Backend log visibility is controlled by `BACKEND_LOGS_VISIBLE` in the local operational environment and defaults to `true` when absent.
- Backend dependency state is locked in `app/server/uv.lock`.
- v3 package dependencies are declared independently in `app/backend/common/pyproject.toml`, `app/backend/core/pyproject.toml`, and `app/backend/ml/pyproject.toml`.

## Constraints

- The repository is Windows-first and uses a PowerShell launcher plus the batch test runner.
- First launch can be slow because runtime binaries and dependencies may need provisioning.
- PostgreSQL must be initialized with the launcher's **Initialize database**
  command before normal application startup; startup never creates or resets
  the external database.
- No container runtime target is currently implemented.
- The Windows launcher currently starts the transitional `app/server` runtime; v3 package launcher integration is not yet implemented.
