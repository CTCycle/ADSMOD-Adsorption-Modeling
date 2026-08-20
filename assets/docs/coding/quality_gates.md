# ADSMOD Quality Gates

Last updated: 2026-08-20

## Python Validation

- Run Ruff across active and extracted Python (`ruff check app/server app/backend app/tests` in CI).
- Test with pytest, including the relevant `app/tests` unit, backend, server, persistence, and E2E scopes.
- Run all extracted-v3 tests from the repository root with explicit source paths:
  `app/backend/common/src`, `app/backend/core/src`, and `app/backend/ml/src`.
- Run the dependency-boundary tests that reject service reversal, framework or
  persistence imports from contracts, retired import paths, and transitional
  dependencies from v3 packages.
- Regenerate `app/resources/adsmod.schema.json` from `AdsmodConfig` and require
  a clean diff on a second generation pass. The JSON schema is generated output,
  not an editable configuration authority.

## Frontend Validation

- Keep `npm run lint`, `npm run test`, and `npm run build` passing for touched frontend behavior.
- Run `npm run visual:compare` when changing layout, responsive behavior, or visual states.

## Cross-Cutting Expectation

- Validate the smallest relevant scope first, then expand only when the change crosses service or frontend boundaries.
- Keep documentation updated when implementation changes alter architecture, runtime behavior, or user workflows.
- Regenerate core, ML, and unified OpenAPI snapshots from running applications;
  snapshots are derived contracts and must not be hand-edited.
