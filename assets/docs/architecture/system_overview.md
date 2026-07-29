# ADSMOD System Overview

Last updated: 2026-07-29

## Platform Shape

ADSMOD is a Windows-first local application with:

- Canonical v3 backend packages under `app/backend`
  - `common` for versioned configuration, health, capability, and error contracts.
  - `core` for the core ASGI application, CLI, and immutable snapshot store.
  - `ml` for the independent ML health/capability service and core snapshot client.
- Transitional backend services under `app/server`
  - `core_service` for non-ML API workflows such as health, datasets, fitting, and NIST.
  - `ml_service` for training datasets, checkpoints, and training lifecycle workflows.
  - `shared` for persistence, repositories, schemas, and common backend utilities.
- One frontend
  - `app/client` for source, fitting, and training workflows.
  - Training remains visible in the unified UI but depends on the optional ML service at runtime.
- Runtime bootstrap assets under `runtimes/`.

Canonical v3 configuration is `app/resources/adsmod.json`, validated by
`app/resources/adsmod.schema.json`. Operational environment toggles remain in
`settings/.env.example`; service-specific JSON configuration is not supported.

## Backend Package Layout

```text
app/server/
  pyproject.toml
  uv.lock

  core_service/
    pyproject.toml
    core_service/
      app.py
      api/
      configurations/
      domain/
      services/
      common/

  ml_service/
    pyproject.toml
    ml_service/
      app.py
      api/
      configurations/
      domain/
      services/
      learning/
      common/

  shared/
    pyproject.toml
    shared/
      repositories/
      persistence/
      models/
      common/

app/backend/
  common/       # adsmod-common
  core/         # adsmod-core
  ml/           # adsmod-ml
```

## Service Entry Points

- Core ASGI app: `core_service.app:app`
- ML ASGI app: `ml_service.app:app`
- Unified local-web backend composition entrypoint: `app.server.app:app`

v3 entrypoints are created through `adsmod_core.create_app_from_path(...)` and
`adsmod_ml.create_app(...)`; the core CLI accepts an explicit `--config` path.

The unified entrypoint composes service routers; it does not own backend business handlers.

## Frontend Responsibility

- `app/client` owns `source`, `fitting`, and `training`.
- `/api/training/*` traffic is routed to `ml_service` in development proxy mode.
- Other `/api/*` traffic is routed to `core_service`.
- In core-only mode, training routes show an unavailable state instead of failing the Source and Fitting workflows.
