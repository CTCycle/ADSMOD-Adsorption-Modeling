# ADSMOD operational commands

Last updated: 2026-10-20

## Launch and maintenance

```powershell
& .\start_on_windows.ps1
```

Use the menu for dependency synchronization, frontend rebuild, database
initialization, log removal, cache cleanup, checkpoint removal, and uninstall.
Runtime host and port values are loaded from SQLite. Database selection,
credentials, and the settings database path are bootstrap values in
`settings/.env`; see [runtime configuration](../runtime/configuration.md).

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

Import settings from a legacy installation once, before removing its old file:

```powershell
Push-Location .\app
& .\server\.venv\Scripts\python.exe -m scripts.import_legacy_configuration --legacy-config ..\data\adsmod.json
Pop-Location
```

The canonical OpenAPI snapshot includes the optional training and checkpoint
routes, so generate it from the ML-enabled development profile:

```powershell
& .\runtimes\uv\uv.exe run --project .\app\server --extra ml --group dev python app\scripts\generate_openapi.py --output app\server\openapi\backend.json
```

The generator fails closed under the Base profile rather than overwriting the
canonical snapshot with a reduced contract.

## Frontend

From `app/client`, run `npm ci`, `npm run dev`, `npm run lint`, `npm run test`,
or `npm run build` as appropriate. The proxy sends training requests before
the general `/api/v1` route.
