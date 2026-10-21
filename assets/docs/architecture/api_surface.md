# API surface

Last updated: 2026-10-21

ADSMOD exposes one FastAPI backend. Liveness and readiness are available at
`/health/live` and `/health/ready`; there are no service-specific health
aliases or unversioned API aliases.

## Versioned API

The unified backend serves all versioned routes under `/api/v1`:

- `/system/capabilities` and `/system/configuration`
- `/datasets/*`
- `/public-data/*` (NIST, PubChem, and COD acquisition and browsing)
- `/fitting/*`
- `/analytics/*` (read-only cross-dataset queries for the dashboards:
  `GET /overview`, and `POST /selection`, `/series`, `/uptake-at-pressure`,
  `/distribution`)
- `/training/configuration` and the training lifecycle under `/training/*`
  when the optional machine learning dependencies are installed

`/system/capabilities` is the authoritative runtime feature-discovery endpoint.
The frontend uses its `features.machine_learning` value to expose or hide
machine learning navigation and routes.

Analytics responses are computed per request from canonical observations and
are never persisted. A response that shares a pressure axis never mixes
pressure bases (a mixed selection returns 400), every plotted point keeps its
`observation_id`, and calculated values carry a `method` (`measured`,
`nearest`, or `interpolated`). Fixed-pressure comparisons interpolate in log
pressure between bracketing points of the adsorption branch (points up to the
pressure maximum in acquisition order) and never extrapolate. Series requests
are capped at 200 series and 50,000 points and report `truncated`.

The canonical OpenAPI snapshot is `app/server/openapi/backend.json`.

The Angular client sends all API requests to the same backend origin. There is
no frontend routing split between core and machine learning services.
