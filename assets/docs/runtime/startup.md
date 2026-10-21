# ADSMOD startup procedures

Last updated: 2026-10-21

## Recommended startup

```powershell
& .\start_on_windows.ps1
```

The launcher reads runtime host and ports through `server.runtime_config`
from the active database. The storage root is resolved first; database
connection settings come from `database.json` inside that root. It inspects every configured application port before doing
runtime, dependency, or build work, then starts one FastAPI backend and a static
production preview server. If configured ports
are occupied by resolvable processes, it presents the complete conflict set
and asks once whether to terminate the deduplicated process list. A decline,
non-interactive console, unresolved owner, ownership race, failed termination,
or final occupied-port check cancels startup without launching ADSMOD. The
launcher never terminates an unresolved listener or a process that no longer
matches the preflight identity.

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

`app/client/scripts/preview-serve.mjs` serves `dist/browser` directly, falls
back to `index.html` for extensionless Angular routes, rejects traversal, and
proxies `/api/v1` and `/health` through the existing `proxy.conf.cjs` target.
It fails immediately when the generated `index.html` is missing, so normal
launches do not invoke Angular compilation a second time. The launcher waits
for `/health/ready` and the frontend root while monitoring each started
process; an early exit is reported immediately instead of waiting for the full
timeout.

After a successful launch, the launcher closes automatically and hands off
ownership. The backend continues in its visible terminal and the frontend
preview continues in the background. To stop both services later, start the
launcher again and use **Kill all application processes**; it stops recognized
ADSMOD backend/frontend process trees after confirmation. **Stop application**
only applies to processes still owned by the current launcher session. That
menu action remains separate from arbitrary port-conflict resolution.

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

## Manual backend startup

From the repository root after `app/server/.venv` is ready:

```powershell
& .\app\server\.venv\Scripts\python.exe -m server.cli
```

This is the only backend process. The storage root and dedicated database
configuration are resolved before connecting; application behavior settings
load from the selected database. If a `database.json` file does not yet exist,
startup performs the one-time legacy configuration cutover. If the environment was synchronized with the
`ml` extra, the process discovers and registers the optional ML extension at
startup. The Angular development server can be started from `app/client` with
`npm run dev`; this is separate from the launcher’s production preview path.

## Database startup rules

The backend runs the synchronous Alembic coordinator before serving requests.
A missing or empty SQLite file is initialized to the packaged head. A non-empty
unversioned file, an empty version table beside application tables, an unknown
revision, or a stamped-but-incomplete schema fails explicitly without schema
inference. PostgreSQL uses the same migration history and a bounded advisory
lock. The active database contains both operational data and the
`application_settings` row. The separate `database.json` file contains only
the selected backend and connection parameters, so it remains readable before
any database is available.

The Database tab can switch the active connection without restarting when the
target connection and schema are ready. A switch waits for ordinary requests
and database sessions to drain, and is rejected while a job or public-data
sync run is active. Each database retains its own settings and operational
data; switching does not copy either. Host and listener port changes still
require a restart because they change process bindings.

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
npm run test:unit
npm run test:preview
npm run build
```
