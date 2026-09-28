# ADSMOD operational commands

Last updated: 2026-09-28

## Launch and maintenance

```powershell
& .\start_on_windows.ps1
```

Use the menu for dependency synchronization, frontend rebuild, database
initialization, log removal, cache cleanup, checkpoint removal, and uninstall.
The launcher reads `resources/adsmod.json` by default. Select an alternate
resource directory when needed:

```powershell
& .\start_on_windows.ps1 -ResourcesPath .\alternate-resources
```

The `ADSMOD_RESOURCES_DIR` environment variable provides the same override.

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
& .\app\server\.venv\Scripts\python.exe app\scripts\generate_config_schema.py --output resources\adsmod.schema.json
```

The canonical OpenAPI snapshot includes the optional training and checkpoint
routes, so generate it from the ML-enabled development profile:

```powershell
& .\runtimes\uv\uv.exe run --project .\app\server --extra ml --group dev python app\scripts\generate_openapi.py --config resources\adsmod.json --output app\server\openapi\backend.json
```

The generator fails closed under the Base profile rather than overwriting the
canonical snapshot with a reduced contract.

## Frontend

From `app/client`, run `npm ci`, `npm run dev`, `npm run lint`, `npm run test`,
or `npm run build` as appropriate. The proxy sends training requests before
the general `/api/v1` route.
