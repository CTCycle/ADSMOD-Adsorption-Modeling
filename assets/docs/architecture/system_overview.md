# ADSMOD System Overview

Last updated: 2026-06-04

## Platform Shape

ADSMOD is a Windows-first local application with:

- Backend services under `app/server`
  - `core_service` for non-ML API workflows such as health, datasets, fitting, and NIST.
  - `ml_service` for training datasets, checkpoints, and training lifecycle workflows.
  - `shared` for persistence, repositories, schemas, and common backend utilities.
- Two frontends
  - `app/client` for source and fitting workflows.
  - `app/ml_client` for training workflows.
- Optional desktop shell
  - Tauri under `app/client/src-tauri` and `release/tauri`.
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
- Unified local-web/Tauri backend composition entrypoint: `app.server.app:app`

The unified entrypoint composes service routers; it does not own backend business handlers.

## Frontend Responsibility Split

- `app/client` owns `source` and `fitting`.
- `app/ml_client` owns `training`.
- The core frontend talks to non-training backend routes.
- The ML frontend talks to `/api/training/*` routes.
