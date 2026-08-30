# ADSMOD local deployment

Last updated: 2026-08-30

The supported deployment is the Windows local web launcher:

```powershell
& .\start_on_windows.ps1
```

The launcher provisions portable Python, uv, and Node.js under `runtimes/`,
synchronizes `app/backend` into `app/backend/.venv`, builds the Angular client,
starts Core, optionally starts ML for `core-ml`, and serves the production
preview on the configured frontend port.

The browser uses same-origin `/api/v1` requests. The Angular proxy sends
`/api/v1/training` to ML and the general `/api/v1` surface to Core. Both
services expose `/health/live` and `/health/ready`.

Embedded SQLite is stored below the configured storage root. PostgreSQL remains
available through the typed Core database configuration; startup applies the
packaged Alembic history and does not infer an unknown schema.

Disposable uv, npm, Python, pytest, and frontend caches belong under
`runtimes/cache` or `app/tests/cache`. No container deployment target is
currently implemented.
