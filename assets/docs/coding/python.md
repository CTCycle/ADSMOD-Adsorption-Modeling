# ADSMOD Python rules

Last updated: 2026-09-16

## Runtime and typing

- Target Python `>=3.14`.
- Use `app/server/.venv` and keep it aligned with `app/server/uv.lock`.
- Annotate public APIs and non-trivial logic with built-in generic types.
- Use Pydantic contracts for request, response, and workflow validation.

## Configuration and services

- `AdsmodConfig` is the only configuration-shape authority.
- `server.repositories` owns SQLAlchemy and operational persistence;
  `server.migrations` owns Alembic history and configuration.
- `server.models` and ML-owned services provide model execution and artifacts;
  they consume immutable snapshots through the shared in-process contract and
  must not import repositories, migrations, or the ORM.
- `server.common` and `server.domain` remain framework-neutral.
- `server.app` and core services must not import Torch, Keras, or scikit-learn;
  the launcher selects the optional `ml` dependency profile explicitly.
- Do not add compatibility imports, alternate config files, or route aliases.

## Long-running work

Do not execute CPU-heavy workloads inside asynchronous request handlers. Expose
explicit start, status/poll, and cancel operations through the owning service's
job system.

## Scientific data

Keep unit aliases and conversion factors in the canonical common unit registry.
Service code may orchestrate DataFrame columns around that registry but must
not duplicate scientific constants.
