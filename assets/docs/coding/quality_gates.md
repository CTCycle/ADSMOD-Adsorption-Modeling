# ADSMOD quality gates

Last updated: 2026-08-30

## Python

- Run Ruff over `app/backend`, `app/tests`, and `app/scripts`.
- Run the backend package tests plus `app/tests/backend`,
  `app/tests/persistence`, and `app/tests/unit`.
- Run the dependency-boundary tests that enforce framework, persistence, and
  Core/ML import direction.
- Regenerate `app/resources/adsmod.schema.json`,
  `app/backend/openapi/core.json`, and `app/backend/openapi/ml.json`; require
  no diff after generation.

## Frontend

For client changes, run `npm run lint`, `npm run test`, and `npm run build` from
`app/client`. Run the visual comparison command when layout, responsive
behavior, or visual states change.

## Cross-cutting

Validate the smallest relevant scope first, then expand when a change crosses
service or frontend boundaries. Distinguish local test results from live
provider, browser, and long-running ML validation.
