# ADSMOD Persistence And Packages

Last updated: 2026-07-15

## Backend Workspace Model

- Shared backend environment: `app/server/.venv`
- Shared backend lockfile: `app/server/uv.lock`
- Root backend workspace definition: `app/server/pyproject.toml`
- Workspace members: `shared`, `core_service`, `ml_service`

## Persistence Ownership

Persistence and data access shared by multiple services live in `app/server/shared/shared`:

- `DatabaseManager` owns the engine, session factory, SQLite pragmas, disposal, and transaction context.
- Typed repositories (`datasets`, `materials`, `isotherms`, `fitting`, and `training`) own explicit conflict targets and SQL projections.
- `schemas/models.py` is the canonical relationship-aware 11-table ORM schema.
- Persistence-safe serializers and shared helpers remain here; SQL and session ownership are being removed from serializers.
- shared infrastructure services that do not depend on `core_service` or `ml_service`

The canonical tables are `datasets`, `adsorbates`, `adsorbents`, `isotherms`,
`isotherm_components`, `isotherm_measurements`, `processed_isotherms`, `fits`,
`fit_parameters`, `training_datasets`, and `training_samples`. Dataset deletion is
a hard delete with database cascades. Public identities are normalized or hashed
in application code, timestamps are UTC-aware, and relationship loading defaults
to `lazy="raise"` so accidental N+1 access fails during development.

The schema and typed repositories are implemented in the current migration slice.
The legacy generic facade and consumer query/serializer paths remain transitional
until their callers are migrated; they are not part of the canonical API.

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
