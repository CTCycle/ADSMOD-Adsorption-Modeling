# ADSMOD Service Boundaries

Last updated: 2026-06-03

## Dependency Direction

- `core_service -> shared`
- `ml_service -> shared`
- `shared -> no service package`

## Prohibited Imports

- `core_service` must not import `ml_service`.
- `shared` must not import `core_service` or `ml_service`.
- `core_service` must not import ML-heavy dependencies such as `torch`, `keras`, or `scikit-learn`.

## Ownership Rules

- ML-heavy dependencies belong only in `app/server/ml_service/pyproject.toml`.
- Shared persistence, CRUD logic, ORM models, and database/session utilities belong in `app/server/shared`.
- ML-specific training and checkpoint implementation stays in `ml_service`.
