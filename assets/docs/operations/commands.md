# ADSMOD operational commands

Last updated: 2026-10-21

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
