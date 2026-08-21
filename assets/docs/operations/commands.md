# ADSMOD Operational Commands

Last updated: 2026-08-21

## Launch And Maintenance

- Unified launcher and maintenance menu: `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1`
- Frontend-only rebuild: select **Rebuild Frontend** in the launcher.
- Database initialization: select **Initialize database** in the launcher. This
  creates missing storage, adopts a validated pre-Alembic schema when needed,
  and upgrades both SQLite and PostgreSQL to the packaged Alembic head. It does
  not reset an existing database.
- Cache cleanup: select **Clear cache** in the launcher. Runtime caches are
  under `runtimes/cache`; pytest and other test-tool caches are under
  `app/tests/cache`. Locked files are skipped with warnings.
- Update status: select **Check for Updates**. This compares local `main` with
  `origin/main` without downloading or applying changes.
- Application update: select **Update**. The launcher switches to `main` and
  runs `git pull --ff-only origin main` after confirming that the working tree
  is clean.
- Checkpoint cleanup: select **Remove Checkpoints** to delete saved training
  checkpoints without removing the database or other local data.
- Full local data cleanup: select **Remove All Data** to delete the embedded
  database, SQLite sidecar files, saved checkpoints, and generated logs while
  preserving application files and settings.

## Alembic Development Workflow

Run these commands from the repository root after dependencies are installed:

```powershell
& .\app\server\.venv\Scripts\python.exe -m alembic --config app\server\pyproject.toml current --check-heads
& .\app\server\.venv\Scripts\python.exe -m alembic --config app\server\pyproject.toml check
& .\app\server\.venv\Scripts\python.exe -m alembic --config app\server\pyproject.toml revision --autogenerate -m "describe schema change"
& .\app\server\.venv\Scripts\python.exe -m alembic --config app\server\pyproject.toml upgrade head
& .\app\server\.venv\Scripts\python.exe -m alembic --config app\server\pyproject.toml history
```

Review every generated revision before applying it, including SQLite batch
operations and any data transformation or constraint change.

## Tests

- Full runner: `app\tests\run_tests.bat`
- Direct pytest: `.\app\server\.venv\Scripts\python.exe -m pytest app\tests -v`
- SQLite persistence contract: `.\app\server\.venv\Scripts\python.exe -m pytest app\tests\persistence -v`

## OpenAPI Snapshots

- Unified schema: `$env:ADSMOD_ENABLE_ML="true"; .\app\server\.venv\Scripts\python.exe -m app.scripts.generate_openapi --app app.server.app:app --output app\shared\openapi.json`

## Frontend Development

- Unified frontend dev server: run `npm run dev` from `app\client`
- Frontend build: run `npm run build` from `app\client`
- Frontend visual comparison: run `npm run visual:compare` from `app\client`
