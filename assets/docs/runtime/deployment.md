# ADSMOD local deployment

Last updated: 2026-09-02

The supported deployment is the Windows local web launcher:

```powershell
& .\start_on_windows.ps1
```

The launcher provisions portable Python, uv, and Node.js under `runtimes/`,
synchronizes `app/server` into `app/server/.venv`, builds the Angular client,
starts one FastAPI backend, and serves the production frontend on the
configured frontend port.

During installation the user can choose the base backend or include the
optional machine learning dependency extra. Both profiles run the same backend
entry point. The ML-enabled profile adds training routes in-process; it does
not start another server.

The browser uses same-origin `/api/v1` requests. The unified backend exposes
`/health/live`, `/health/ready`, and `/api/v1/system/capabilities`. The frontend
uses the capabilities response to decide whether ML functionality is
available.

Embedded SQLite is stored below the configured storage root. PostgreSQL remains
available through the typed database configuration; startup applies the
packaged Alembic history and does not infer an unknown schema.

Disposable uv, npm, Python, pytest, and frontend caches belong under
`runtimes/cache` or `app/tests/cache`. No container deployment target is
currently implemented.
