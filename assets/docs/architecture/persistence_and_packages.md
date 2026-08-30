# Persistence and packages

Last updated: 2026-08-30

## Backend workspace

`app/backend/pyproject.toml` and `app/backend/uv.lock` define one workspace
with the `adsmod-common`, `adsmod-core`, and `adsmod-ml` packages. The launcher
and CI install from this workspace and use its lockfile.

## Core persistence

Core owns the operational database and Alembic history under
`app/backend/migrations`. Startup accepts only an empty unversioned SQLite
file or a known Alembic state; it never infers or silently adopts an unknown
schema. The current schema includes immutable `training_snapshots` and
`training_snapshot_rows` tables.

## ML artifacts

ML does not open the operational database. It requests snapshots from Core,
checks the returned hash, and keeps its training manifest and checkpoints under
the configured storage root. This prevents ORM changes from becoming an
implicit ML API.
