# ADSMOD startup procedures

Last updated: 2026-09-21

## Recommended startup

```powershell
& .\start_on_windows.ps1
```

The launcher reads `app/resources/adsmod.json`, inspects every configured
application port before doing runtime, dependency, or build work, then starts
one FastAPI backend and a static production preview server. If configured ports
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

After a successful launch, use **Stop application** in the same launcher
session to stop only the backend and frontend processes started by that
session. The backend runs in a visible terminal when launched from the
PowerShell script. If an earlier session left ADSMOD processes running, use
**Kill all application processes**; it stops recognized ADSMOD
backend/frontend process trees after confirmation. That menu action remains
separate from arbitrary port-conflict resolution.

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
& .\app\server\.venv\Scripts\python.exe -m server.cli --config .\app\resources\adsmod.json
```

This is the only backend process. If the environment was synchronized with the
`ml` extra, the process discovers and registers the optional ML extension at
startup. The Angular development server can be started from `app/client` with
`npm run dev`; this is separate from the launcher’s production preview path.

## Database startup rules

The backend runs the synchronous Alembic coordinator before serving requests.
A missing or empty SQLite file is initialized to the packaged head. A non-empty
unversioned file, an empty version table beside application tables, an unknown
revision, or a stamped-but-incomplete schema fails explicitly without schema
inference. PostgreSQL uses the same migration history and a bounded advisory
lock.

## Tests

```cmd
app\tests\run_tests.bat
```

The automated suite validates configuration, persistence, backend routes,
frontend behavior, and both dependency profiles. Live browser and
hardware-specific ML checks should be run locally on the target machine.

The frontend checks can also be run directly from `app/client`:

```cmd
npm run lint
npm run test:unit
npm run test:preview
npm run build
```
