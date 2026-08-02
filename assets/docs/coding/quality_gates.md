# ADSMOD Quality Gates

Last updated: 2026-08-02

## Python Validation

- Run Ruff through the backend workspace (`ruff check app/server` in CI).
- Test with pytest, including the relevant `app/tests` unit, backend, server, persistence, and E2E scopes.

## Frontend Validation

- Keep `npm run lint`, `npm run test`, and `npm run build` passing for touched frontend behavior.
- Run `npm run visual:compare` when changing layout, responsive behavior, or visual states.

## Cross-Cutting Expectation

- Validate the smallest relevant scope first, then expand only when the change crosses service or frontend boundaries.
- Keep documentation updated when implementation changes alter architecture, runtime behavior, or user workflows.
