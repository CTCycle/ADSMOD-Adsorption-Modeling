# ADSMOD quality gates

Last updated: 2026-09-02

## Python

- Run Ruff over `app/backend`, `app/tests`, and `app/scripts`.
- Run the backend package tests plus `app/tests/backend`,
  `app/tests/persistence`, and `app/tests/unit`.
- Run dependency-boundary tests that enforce framework, persistence, and
  optional-extension import direction.
- Validate both dependency profiles: base backend without ML and backend with
  the `ml` extra installed.
- Regenerate `app/resources/adsmod.schema.json` and
  `app/backend/openapi/backend.json`; require no unexpected diff after
  generation.

## Frontend

For client changes, run `npm run lint`, `npm run test:unit`, and
`npm run build` from `app/client`. Run visual comparison tests when layout,
responsive behavior, or visual states change.

Capability-sensitive tests must cover both `features.machine_learning = false`
and `features.machine_learning = true` so the frontend does not expose an
optional feature that the backend cannot provide.

## Cross-cutting

Validate the smallest relevant scope first, then expand when a change crosses
backend, optional-extension, or frontend boundaries. Distinguish automated
checks from live browser, hardware-specific, and long-running ML validation.
