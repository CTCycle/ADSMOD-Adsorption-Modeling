# ADSMOD Persistence And Packages

Last updated: 2026-07-09

## Backend Workspace Model

- Shared backend environment: `app/server/.venv`
- Shared backend lockfile: `app/server/uv.lock`
- Root backend workspace definition: `app/server/pyproject.toml`
- Workspace members: `shared`, `core_service`, `ml_service`

## Persistence Ownership

Persistence and data access shared by multiple services live in `app/server/shared/shared`:

- database backend and session utilities
- repository queries
- ORM schemas and models
- persistence-safe serializers and shared helpers
- shared infrastructure services that do not depend on `core_service` or `ml_service`

ML-specific model and checkpoint serialization remains under `ml_service`.

## Validation Expectations

Core-only installation uses:

- `uv sync --package adsmod-core-service --group dev` in `app/server`
- import checks for `core_service.app`
- dependency-boundary checks for `core_service` and `shared`

ML-enabled installation uses:

- `uv sync --package adsmod-ml-service --group dev` for ML-only work
- `uv sync --all-packages --group dev` only when both services are explicitly requested
- import checks for `ml_service.app` only in ML-enabled validation

Architecture validation requires:

- route-separation checks for training endpoints
- backend tests and generated OpenAPI artifacts for both services
