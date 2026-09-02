# ADSMOD operational commands

Last updated: 2026-09-02

## Launch and maintenance

```powershell
& .\start_on_windows.ps1
```

Use the menu for dependency synchronization, frontend rebuild, database
initialization, log removal, cache cleanup, checkpoint removal, and uninstall.
The launcher always reads `app/resources/adsmod.json`.

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
& .\app\server\.venv\Scripts\python.exe -m pytest -c app\tests\pytest.ini app\tests -v --basetemp app\tests\cache\pytest-tmp-local
& .\app\server\.venv\Scripts\python.exe app\scripts\generate_openapi.py --config app\resources\adsmod.json --output app\server\openapi\backend.json
& .\app\server\.venv\Scripts\python.exe app\scripts\generate_config_schema.py --output app\resources\adsmod.schema.json
```

## Frontend

From `app/client`, run `npm ci`, `npm run dev`, `npm run lint`, `npm run test`,
or `npm run build` as appropriate. The proxy sends training requests before
the general `/api/v1` route.
