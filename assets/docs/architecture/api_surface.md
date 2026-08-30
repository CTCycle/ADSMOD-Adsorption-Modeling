# API surface

Last updated: 2026-08-30

Both services expose liveness and readiness at `/health/live` and
`/health/ready`. There are no unversioned API aliases.

## Core service

Core serves the versioned routes under `/api/v1`:

- `/system/capabilities` and `/system/configuration`
- `/datasets/*`
- `/nist/*`
- `/fitting/*`
- `/internal/snapshots/*` and `/internal/training/*` for authenticated ML
  coordination

The Core OpenAPI snapshot is `app/backend/openapi/core.json`.

## ML service

ML serves `/api/v1/system/capabilities`,
`/api/v1/training/configuration`, and the training lifecycle under
`/api/v1/training/*`. The ML OpenAPI snapshot is
`app/backend/openapi/ml.json`.

The frontend proxy matches `/api/v1/training` before the general `/api/v1`
route so training requests reach ML and all other API requests reach Core.
