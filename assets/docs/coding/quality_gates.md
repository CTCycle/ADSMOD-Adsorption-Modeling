# ADSMOD Quality Gates

Last updated: 2026-06-03

## Python Validation

- Lint and format with Ruff, or the project-standard formatter and linter when configured.
- Type-check with Pylance.
- Test with pytest, including relevant `tests/unit` and `tests/e2e` or `tests/server` coverage for changed behavior.

## Frontend Validation

- Keep `npm run build` passing for touched frontend applications.
- Update frontend tests when behavior changes.

## Cross-Cutting Expectation

- Validate the smallest relevant scope first, then expand only when the change crosses service or frontend boundaries.
- Keep documentation updated when implementation changes alter architecture, runtime behavior, or user workflows.
