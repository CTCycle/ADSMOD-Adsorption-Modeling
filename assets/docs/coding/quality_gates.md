# ADSMOD quality gates

Last updated: 2026-10-22

## Python

- Run Ruff over `app/server`, `app/tests`, and `app/scripts`.
- Run the backend package tests plus `app/tests/backend`,
  `app/tests/persistence`, and `app/tests/unit`.
- Run dependency-boundary tests that enforce framework, persistence, and
  optional-extension import direction.
- Validate both dependency profiles: base backend without ML and backend with
  the `ml` extra installed.
- Regenerate `app/server/openapi/backend.json`; require no unexpected diff
  after generation. The canonical OpenAPI snapshot must be generated from the
  ML-enabled profile because it includes training and checkpoint routes; the
  generator fails closed under the Base profile.

## Frontend

For client changes, run `npm run lint`, `npm test`, and `npm run build` from
`app/client`. Run `npm run browser:validate` when layout, responsive behavior,
or browser interactions change. The browser suite checks four representative
viewports: 1440px, 1280px, 900px, and 600px; tests can set narrower widths for
specific boundary assertions.

Capability-sensitive tests must cover both `features.machine_learning = false`
and `features.machine_learning = true` so the frontend does not expose an
optional feature that the backend cannot provide.

## Cross-cutting

Validate the smallest relevant scope first, then expand when a change crosses
backend, optional-extension, or frontend boundaries. Distinguish automated
checks from live browser, hardware-specific, and long-running ML validation.
For application settings, verify persistence in the active database,
full-model validation, runtime application, restart indicators, and the actual
Settings page at desktop and narrow widths. For database configuration,
verify protected file persistence, connection testing, and switching or
rollback behavior for the available backend profile.
