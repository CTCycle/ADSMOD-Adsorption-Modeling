# ADSMOD operational commands

Last updated: 2026-10-29

## Launch and maintenance

```powershell
& .\start_on_windows.ps1
```

Use the menu for dependency synchronization, frontend rebuild, database
initialization, log removal, cache cleanup, checkpoint removal, and uninstall.
Runtime host, ports, and CORS values are loaded from `settings/.env` or the
process environment, with process values taking precedence. Database selection
and connection values are loaded from `database.json` under the configured data
root. `settings/.env` may also set `ADSMOD_STORAGE_ROOT`; see
[runtime configuration](../runtime/configuration.md).

Preview a cleanup action without deleting files:

```powershell
& .\start_on_windows.ps1 -DryRun
```

Choose **Remove logs**, **Clear cache**, **Remove checkpoints**, **Remove all
data**, or **Uninstall application** in the menu. Dry-run mode previews only
these cleanup actions; it does not run launch, install, update, or build actions.
The launcher resolves the same configured storage root as the backend without
installing runtimes. QA folders are excluded from cleanup.

**Clear cache** removes recognized runtime/tool caches, nested Python and
frontend caches, known test temporary folders, interrupted launcher downloads,
backend and frontend coverage output, TypeScript compiler output, and the
generated frontend build. **Uninstall application**
also removes portable runtimes, Python environments, frontend dependencies, and
application logs, while preserving database configuration, the selected local
database, checkpoints, and training artifacts. **Remove all data** is the
separate persistent-data action; it lists the configured database, sidecars,
database configuration, logs, checkpoints, and training artifacts before
confirmation. A configured PostgreSQL database is reported as external and is
never modified.

## Backend workspace

```powershell
& .\runtimes\uv\uv.exe sync --locked --project .\app\server --all-packages --group dev
```

Alembic commands run with the backend environment and package-local config:

```powershell
& .\app\server\.venv\Scripts\python.exe -m alembic --config .\app\server\pyproject.toml current --check-heads
& .\app\server\.venv\Scripts\python.exe -m alembic --config .\app\server\pyproject.toml check
& .\app\server\.venv\Scripts\python.exe -m alembic --config .\app\server\pyproject.toml upgrade head
```

## Tests and schemas

```cmd
app\tests\run_tests.bat
```

```powershell
& .\app\server\.venv\Scripts\python.exe -m pytest -c app\tests\pytest.ini app\tests -v --basetemp .\runtimes\cache\pytest-tmp
```

The current schema is created on first launch when the selected database is
empty. Database files stamped with historical revisions are not upgraded; create
a fresh database and select it through the Database settings page. Startup does
not remove or recreate existing database files.

The canonical OpenAPI snapshot includes the optional training and checkpoint
routes, so generate it from the ML-enabled development profile:

```powershell
& .\runtimes\uv\uv.exe run --project .\app\server --extra ml --group dev python app\scripts\generate_openapi.py --output app\server\openapi\backend.json
```

The generator fails closed under the Base profile rather than overwriting the
canonical snapshot with a reduced contract.

## Frontend

From `app/client`, run `npm ci`, `npm run dev`, `npm run lint`, `npm test`,
`npm run build`, and `npm run browser:validate` as appropriate. The proxy sends
training requests before the general `/api/v1` route.
