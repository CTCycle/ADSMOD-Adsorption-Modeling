# ADSMOD Python rules

Last updated: 2026-09-03

## Runtime and typing

- Target Python `>=3.14`.
- Use `app/server/.venv` and keep it aligned with `app/server/uv.lock`.
- Annotate public APIs and non-trivial logic with built-in generic types.
- Use Pydantic contracts for request, response, and workflow validation.

## Configuration and services

- `AdsmodConfig` is the only configuration-shape authority.
- Core owns SQLAlchemy, Alembic, repositories, and operational persistence.
- ML owns model execution and artifacts and consumes immutable Core-owned snapshots through the shared in-process contract; it must not import Core persistence packages or the ORM.
- `adsmod_common` remains framework-neutral.
- Do not add compatibility imports, alternate config files, or route aliases.

## Long-running work

Do not execute CPU-heavy workloads inside asynchronous request handlers. Expose
explicit start, status/poll, and cancel operations through the owning service's
job system.

## Scientific data

Keep unit aliases and conversion factors in the canonical common unit registry.
Service code may orchestrate DataFrame columns around that registry but must
not duplicate scientific constants.
