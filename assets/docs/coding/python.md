# ADSMOD Python Rules

Last updated: 2026-08-20

## Runtime And Environment

- Target Python version: `>=3.14` from `pyproject.toml`.
- Use the backend workspace environment at `app/server/.venv`.
- Do not create new virtual environments for normal development.
- Keep dependency state aligned with `uv` and `app/server/uv.lock`.
- The current backend workspace packages are versioned `3.0.0`; the extracted
  v3 packages under `app/backend` use the same Python floor and independent
  package manifests.

## Typing

- Type annotations are required for public APIs and non-trivial logic.
- Use built-in generics such as `list[str]` and `dict[str, Any]`.
- Prefer `A | B` over `typing.Union`.
- Prefer `collections.abc` for abstract contracts such as `Callable` and `Iterable`.
- Treat typing as a quality requirement, not optional documentation.

## Validation And API Contracts

- Use Pydantic contracts for request, response, and workflow validation.
- Avoid ad-hoc manual validation for payload shapes already represented by models.
- Return explicit HTTP status codes.
- Keep response models consistent and stable.
- Ensure error payloads are safe for clients and traceable through logs or job state.
- `adsmod_common.config.AdsmodConfig` is the only configuration-shape authority;
  `shared.common.settings` may only project its validated values into runtime
  dataclasses.

## Async And Long-Running Work

- Use async handlers only when dependencies are non-blocking.
- Do not run CPU-heavy workloads directly inside async request handlers.
- Route long-running tasks through the service job systems in `core_service/services/jobs.py` and `ml_service/services/jobs.py`.
- Long-running features must expose start, poll or status, and cancel operations.

## Code Structure

- Keep functions focused and small.
- Make side effects explicit.
- Prefer simple composable logic over implicit control flow.
- Add comments only when needed for clarity or safety.
- Follow local style in touched files.
- Keep modules around or below roughly 1000 LOC when practical.
- Keep imports at the file top.
- Avoid nested function definitions unless strict locality is necessary.
- Use classes when they improve cohesion.
- `core_service/contracts`, `ml_service/contracts`, and `shared/contracts` are
  transport/workflow contract packages. They are not nominal domain entities,
  ORM models, or controller modules.
- `shared.services.units.UnitRegistry` is the single source for scientific unit
  aliases, factors, parsing, and conversion. ML conversion code orchestrates
  DataFrame columns around it and must not duplicate factors.

## Service Boundary Rules

- `core_service` may depend on `shared`, but not on `ml_service`.
- `shared` must not depend on either service package.
- `core_service` must not import ML-heavy libraries such as `torch`, `keras`, or `scikit-learn`.
- The v3 packages under `app/backend` must not import transitional service or
  shared packages. Do not add compatibility re-exports or old import paths.
