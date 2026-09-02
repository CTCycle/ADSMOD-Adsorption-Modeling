# API surface

Last updated: 2026-09-02

ADSMOD exposes one FastAPI backend. Liveness and readiness are available at
`/health/live` and `/health/ready`; there are no service-specific health
aliases or unversioned API aliases.

## Versioned API

The unified backend serves all versioned routes under `/api/v1`:

- `/system/capabilities` and `/system/configuration`
- `/datasets/*`
- `/nist/*`
- `/fitting/*`
- `/training/configuration` and the training lifecycle under `/training/*`
  when the optional machine learning dependencies are installed

`/system/capabilities` is the authoritative runtime feature-discovery endpoint.
The frontend uses its `features.machine_learning` value to expose or hide
machine learning navigation and routes.

The canonical OpenAPI snapshot is `app/backend/openapi/backend.json`.

The Angular client sends all API requests to the same backend origin. There is
no frontend routing split between core and machine learning services.
