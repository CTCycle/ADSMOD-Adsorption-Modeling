# ADSMOD System Overview

Last updated: 2026-07-11

## Platform Shape

ADSMOD is a Windows-first local application with:

- Backend services under `app/server`
  - `core_service` for non-ML API workflows such as health, datasets, fitting, and NIST.
  - `ml_service` for training datasets, checkpoints, and training lifecycle workflows.
  - `shared` for persistence, repositories, schemas, and common backend utilities.
- One frontend
  - `app/client` for source, fitting, and training workflows.
  - Training remains visible in the unified UI but depends on the optional ML service at runtime.
- Runtime bootstrap assets under `runtimes/`.

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
```

## Service Entry Points

- Core ASGI app: `core_service.app:app`
- ML ASGI app: `ml_service.app:app`
- Unified local-web backend composition entrypoint: `app.server.app:app`

The unified entrypoint composes service routers; it does not own backend business handlers.

## Frontend Responsibility

- `app/client` owns `source`, `fitting`, and `training`.
- `/api/training/*` traffic is routed to `ml_service` in development proxy mode.
- Other `/api/*` traffic is routed to `core_service`.
- In core-only mode, training routes show an unavailable state instead of failing the Source and Fitting workflows.
