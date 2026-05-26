# ADSMOD Architecture

Last updated: 2026-05-26

## 1. System Overview

ADSMOD is a Windows-first local application with:

- Backend split services under `app/server`:
  - `core_service`: non-ML API workflows (health, datasets, fitting, NIST).
  - `ml_service`: ML API workflows (training datasets, checkpoints, training lifecycle).
  - `shared`: persistence, repositories, schemas, and common backend utilities.
- Frontend split:
  - `app/client`: core frontend (source + fitting) for `core_service`.
  - `app/ml_client`: ML frontend (training) for `ml_service`.
- Optional desktop shell: Tauri under `app/client/src-tauri` and `release/tauri`.
- Runtime/toolchain bootstrap binaries under `runtimes/`.

## 2. Backend Boundary Rules

Dependency direction:

- `core_service -> shared`
- `ml_service -> shared`
- `shared -> no service package`

Prohibited imports:

- `core_service` must not import `ml_service`.
- `shared` must not import `core_service` or `ml_service`.
- `core_service` must not import ML-heavy dependencies (`torch`, `keras`, `scikit-learn`).

## 3. Backend Package Layout

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
      schemas/
      common/
```

## 4. Service Entry Points

- Core ASGI app: `core_service.app:app`
- ML ASGI app: `ml_service.app:app`
- Compatibility shim: `app/server/app.py` re-exports `core_service.app:app` for temporary legacy references only.

## 5. Runtime and Environment Model

- Shared backend environment: `app/server/.venv`
- Shared backend lockfile: `app/server/uv.lock`
- Root backend workspace definition: `app/server/pyproject.toml`
- Workspace members: `shared`, `core_service`, `ml_service`

## 6. API Boundary

Core service must expose non-ML routes only:

- health/root routes
- dataset upload/read (non-training-only)
- fitting routes
- NIST/source collection routes

Core service must not expose `/api/training/*`.

ML service owns training workflows:

- `/api/training/datasets`
- `/api/training/dataset-sources`
- `/api/training/dataset-source`
- `/api/training/build-dataset`
- `/api/training/processed-datasets`
- `/api/training/dataset-info`
- `/api/training/dataset`
- `/api/training/jobs`
- `/api/training/jobs/{job_id}`
- `/api/training/checkpoints`
- `/api/training/checkpoints/{checkpoint_name}`
- `/api/training/start`
- `/api/training/resume`
- `/api/training/stop`
- `/api/training/status`

## 7. Persistence Layer Ownership

Persistence and data access shared by services live in `app/server/shared/shared`:

- database backend/session utilities
- repository queries
- ORM schemas/models
- shared persistence-safe serializers/utilities

ML-specific model/checkpoint serialization remains under `ml_service`.

## 8. CI/Validation Expectations

Stage 1 validation requires:

- `uv sync --all-packages --group dev` in `app/server`
- imports for both `core_service.app` and `ml_service.app`
- dependency boundary checks for `core_service` and `shared`
- route separation checks for training endpoints
- backend tests and generated OpenAPI artifacts for both services
