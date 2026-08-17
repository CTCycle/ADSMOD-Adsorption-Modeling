# ADSMOD Service Boundaries

Last updated: 2026-08-16

## Dependency Direction

- `core_service -> shared`
- `ml_service -> shared`
- `shared -> no service package`

## Prohibited Imports

- `core_service` must not import `ml_service` directly. The unified composition
  entrypoint may conditionally register ML routes when explicitly enabled.
- `shared` must not import `core_service` or `ml_service`.
- `core_service` must not import ML-heavy dependencies such as `torch`, `keras`, or `scikit-learn`.

## Ownership Rules

- ML-heavy dependencies belong only in `app/server/ml_service/pyproject.toml`.
- Shared persistence, CRUD logic, ORM models, and database/session utilities belong in `app/server/shared`.
- `shared.repositories.nist.NISTRepository` is the sole owner of canonical NIST SQLAlchemy reads and writes. The shared `repositories/queries` package contains training-specific query helpers only.
- NIST provider-frame conversion, including unit conversion and canonical record construction, belongs to `core_service.services.data.nist_mapper.NISTCanonicalMapper`.
- Shared infrastructure that is not business-domain specific, such as reusable job orchestration, belongs in `shared`.
- ML-specific training and checkpoint implementation stays in `ml_service`.
- Legacy monolith paths under `app/server/api`, `common`, `configurations`, `domain`, `learning`, `repositories`, and `services` are not part of the active architecture.
