# ADSMOD startup procedures

Last updated: 2026-10-29

## Recommended startup

```powershell
& .\start_on_windows.ps1
```

The launcher reads runtime host, ports, and CORS configuration through
`server.runtime_config` from `settings/.env` or the process environment; process
values take precedence. The storage root is resolved first; database connection
settings come from `database.json` inside that root. It inspects the backend
port before doing runtime, dependency, or build work, then starts one FastAPI
process that serves both the API and the built Angular bundle on the same
origin. If the backend port is occupied by a resolvable process, it presents the
conflict and asks once whether to terminate the deduplicated process list. A
decline, non-interactive console, unresolved owner, ownership race, failed
termination, or final occupied-port check cancels startup without launching
ADSMOD. The launcher never terminates an unresolved listener or a process that
no longer matches the preflight identity.

The backend and frontend installations carry generated dependency-state
manifests inside `.venv` and `node_modules`. They fingerprint the locked
dependency inputs and pinned Python/Node runtimes, and record the selected
Standard/Development and Base/ML profiles. A matching manifest skips
installation; a stale manifest repairs only that environment while preserving
the previously recorded profile. The frontend build carries
`dist/.adsmod-build-state.json`, whose SHA-256 fingerprint covers the exact
Angular/package/source/public inputs and the pinned Node version. A matching
build is reused; a missing or stale state is rebuilt once. There is no
unconditional rebuild flag.

The backend serves `app/client/dist/browser` itself
(`server/api/frontend.py`): files that exist are returned directly, extensionless
paths fall back to `index.html` for Angular client-side routes, and unknown
`/api/`, `/health/` and missing-asset paths return 404. When no bundle exists the
root redirects to `/docs`. Browser code calls the relative `/api/v1`, so no
proxy or CORS is involved in production. The launcher waits for `/health/ready`
while monitoring the backend process; an early exit is reported immediately
instead of waiting for the full timeout, then opens the backend URL.

The launcher is the lifecycle owner. The backend runs in the launcher's own
console, so its logs stay visible and the file logs under the storage root are
unchanged. Press **Ctrl+C** in that console to stop: the backend runs its normal
uvicorn/lifespan shutdown, the launcher waits up to ten seconds, then stops the
launcher-owned process tree and returns to the menu. If the backend exits on its
own, the launcher reports its exit code and returns to the menu. Closing the
console window ends the backend with it. **Kill all application processes**
remains available for recognized ADSMOD backend trees left by a force-killed
launcher; it stops only those after confirmation and is separate from arbitrary
port-conflict resolution.

The launcher requires PowerShell 7 (`pwsh`); Windows PowerShell 5.1 lacks APIs
used by the build fingerprint.

The interactive menu is generated from structured rows. Its order is
`APPLICATION`, `SETUP & VALIDATION`, `SOURCE CONTROL` (Check before Update),
`DATA & MAINTENANCE`, and a final sequential `EXIT` option. The launcher
computes the numeric-column width from the menu size, so one- and two-digit
options remain aligned. Cleanup actions use a shared target inventory and
`-DryRun` previews targets without changing files. Cache cleanup covers the
launcher cache tree, nested Python and Angular/Vitest caches, test temporaries,
coverage and compiler output, and the generated frontend build. QA folders are
excluded. Standard uninstall removes application runtimes, dependencies,
generated output, and logs while preserving the configured database and training
data; **Remove all data** separately previews and removes the known local data
paths. Reparse points are skipped, and locked, inaccessible, or failed paths
are reported.

## Source updates

Choose **Update** in the launcher menu to update the repository from
`origin/main`. The checkout must be non-detached, clean, and already on
`main`; the launcher runs `git pull --ff-only origin main` and does not switch
branches or modify local changes.

## Manual backend startup

From the repository root after `app/server/.venv` is ready:

```powershell
& .\app\server\.venv\Scripts\python.exe -m server.cli
```

This is the only backend process. The storage root and dedicated database
configuration are resolved before connecting; application behavior settings
load from the selected database. If a `database.json` file does not yet exist,
startup selects the current default SQLite configuration and writes the file after
the fresh database and settings row initialize. Retired environment variables
and old configuration files are not imported. If the environment was synchronized with the
`ml` extra, the process discovers and registers the optional ML extension at
startup. The Angular development server can be started from `app/client` with
`npm run dev` on the configured frontend port, proxying `/api/v1` and `/health`
to the backend; it is a development workflow only. The backend serves the
built bundle from `app/client/dist/browser` when it exists.

## Database startup rules

The backend runs the synchronous Alembic coordinator before serving requests.
A missing or empty SQLite file is initialized by the single current-schema root
revision. A non-empty unversioned file, an empty version table beside application
tables, a historical or unknown revision, or a stamped-but-incomplete schema
fails explicitly without schema inference. Databases with historical revisions
must be recreated; startup does not drop them. PostgreSQL uses the same baseline and a bounded advisory
lock. The active database contains both operational data and the
`application_settings` row. The separate `database.json` file contains only
the selected backend and connection parameters, so it remains readable before
any database is available.

The Database tab can switch the active connection without restarting when the
target connection and schema are ready. A switch waits for ordinary requests
and database sessions to drain, and is rejected while a job or public-data
sync run is active. Each database retains its own settings and operational
data; switching does not copy either. Changes to runtime environment values
require a restart because they change process bindings and middleware configuration.

## Tests

```cmd
app\tests\run_tests.bat
```

The automated suite validates configuration, persistence, backend routes,
frontend behavior, and both dependency profiles. Live browser and
hardware-specific ML checks should be run locally on the target machine.

`app\tests\run_tests.bat` reads the launcher dependency-state manifest and
selects the recorded `Base` or `ML` scope. Base runs intentionally exclude
the ML-only unit and E2E modules; they do not install optional ML packages.
Set `STANDARD_TEST_PROFILE=Base` or `STANDARD_TEST_PROFILE=ML` to override the
detected feature profile explicitly. The comprehensive runner requires a
Development installation; its full ML scope therefore requires Development/ML.

The frontend checks can also be run directly from `app/client`:

```cmd
npm run lint
npm test
npm run build
```
