# System overview

Last updated: 2026-10-20

## Repository layout

```text
settings/   database and path bootstrap template
app/
  server/
    api/                      FastAPI routers and HTTP entrypoints
    common/                   framework-neutral shared helpers
    configurations/           typed settings, SQLite persistence, bootstrap
    domain/                   transport and workflow contracts
    models/                   optional ML model and training primitives
    repositories/             operational persistence and snapshots
    services/                 core workflows and lazy ML services
    migrations/               Alembic package-local history and configuration
    app.py                    explicit FastAPI factory
    cli.py                    command-line entrypoint
    openapi/backend.json      canonical generated API contract
    pyproject.toml            single Hatch package
    uv.lock
  client/      Angular application
  scripts/     maintenance and migration entry points
  tests/       Python, browser, and integration validation
```

## Runtime flow

The Windows launcher reads runtime endpoints and the settings database path
through `server.runtime_config`, which loads bootstrap values and the SQLite
settings row. It synchronizes the backend workspace, starts one FastAPI backend on the configured backend port,
and serves the Angular bundle on the configured frontend port. The browser
uses the same backend for datasets, NIST data, fitting, capability discovery,
and optional machine learning operations.

The `server.api` layer owns HTTP routing, `server.services` owns workflows,
and `server.repositories` owns database-backed data and immutable snapshots.
When the ML extra is installed, the unified application loads the optional ML
services and provides in-process training-data access. No second backend
process, internal HTTP service token, or service-to-service proxy is required.

## Optional machine learning

The base backend installation does not install the heavy ML dependencies. The
optional `ml` dependency extra installs them. At startup the backend lazily
attempts to load `server.services.ml_container` and `server.api.training`, then
reports the result via `/api/v1/system/capabilities`. The Angular client uses
that capability as its single source of truth for ML navigation and route
access.

## Configuration flow

`UserSettings` validates the complete application settings model. The console
stores it in the local SQLite `application_settings` table. `AdsmodConfig`
assembles those values with the database bootstrap model. PostgreSQL selection
affects operational data only; settings remain in local SQLite. Normal startup,
the launcher, and the frontend proxy do not read a runtime JSON settings file.
