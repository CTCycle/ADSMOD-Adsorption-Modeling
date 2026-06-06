# ADSMOD Persistence And Packages

Last updated: 2026-06-04

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

Stage 1 architecture validation requires:

- `uv sync --all-packages --group dev` in `app/server`
- import checks for `core_service.app` and `ml_service.app`
- dependency-boundary checks for `core_service` and `shared`
- route-separation checks for training endpoints
- backend tests and generated OpenAPI artifacts for both services
