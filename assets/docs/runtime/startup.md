# ADSMOD startup procedures

Last updated: 2026-09-01

## Recommended startup

```powershell
& .\start_on_windows.ps1
```

The launcher reads `app/resources/adsmod.json`, synchronizes the locked
`app/backend` workspace, builds the client, starts Core, and waits for
`/health/ready` before opening the browser. ML is started only when
`runtime.mode` is `core-ml`; it receives the same config path and uses
`runtime.ml_port`.

The interactive menu is generated from structured rows. Its order is
`APPLICATION`, `SETUP & VALIDATION`, `SOURCE CONTROL` (Check before Update),
`DATA & MAINTENANCE`, and a final sequential `EXIT` option. The launcher
computes the numeric-column width from the menu size, so one- and two-digit
options remain aligned. Recursive cleanup inventories entries, removes them
deepest-first, preserves required sentinels, and reports locked or inaccessible
paths without masking the original action error.

## Source updates

Choose **Update** in the launcher menu to update the repository from
`origin/main`. The checkout must be non-detached, clean, and already on
`main`; the launcher runs `git pull --ff-only origin main` and does not switch
branches or modify local changes.

## Manual service startup

From the repository root after `app/backend/.venv` is ready:

```powershell
& .\app\backend\.venv\Scripts\python.exe -m adsmod_core.cli --config .\app\resources\adsmod.json
```

For `core-ml` mode, start ML in another terminal:

```powershell
& .\app\backend\.venv\Scripts\python.exe -m adsmod_ml.cli --config .\app\resources\adsmod.json
```

The Angular development server can be started from `app/client` with
`npm run dev`; its proxy reads the fixed `app/resources/adsmod.json` path.

## Database startup rules

Core runs the synchronous Alembic coordinator before serving requests. A
missing or empty SQLite file is initialized to the packaged head. A non-empty
unversioned file, an empty version table beside application tables, an unknown
revision, or a stamped-but-incomplete schema fails explicitly without schema
inference. PostgreSQL uses the same migration history and a bounded advisory
lock.

## Tests

```cmd
app\tests\run_tests.bat
```

The runner uses the canonical ports and does not accept alternate API or
resource-root environment overrides.
