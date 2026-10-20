# Persistence and packages

Last updated: 2026-10-20

## Backend package

`app/server/pyproject.toml` and `app/server/uv.lock` define one Hatch package
named `adsmod-backend-workspace`. The package is rooted at `app/server`, with
the `ml` extra selecting the optional learning dependencies. The launcher and
CI install this package and use its lockfile.

## Core persistence

Core owns the operational database and Alembic history under
`app/server/migrations`. Startup accepts only an empty unversioned SQLite
file or a known Alembic state; it never infers or silently adopts an unknown
schema. The current schema includes immutable `training_snapshots` and
`training_snapshot_rows` tables.

SQLite migrations run with foreign-key enforcement disabled and must pass
`PRAGMA foreign_key_check` before commit. Alembic batch operations rebuild a
SQLite table by dropping the original, and with enforcement on that drop would
cascade into child rows.

SQLite transactions serialize writes within the single backend process to
avoid lock-upgrade collisions when independent public-data runs are queued or
updated concurrently. PostgreSQL continues to use database-managed transaction
concurrency.

## ML artifacts

ML does not open the operational database. It receives snapshots through the
in-process `TrainingDataAccess` contract, checks the content hash, and keeps its
training manifest and checkpoints under the configured storage root. This
prevents ORM changes from becoming an implicit ML API.
