# Persistence and packages

Last updated: 2026-10-21

## Backend package

`app/server/pyproject.toml` and `app/server/uv.lock` define one Hatch package
named `adsmod-backend-workspace`. The package is rooted at `app/server`, with
the `ml` extra selecting the optional learning dependencies. The launcher and
CI install this package and use its lockfile.

## Core persistence

Core owns the active database and its Alembic baseline under
`app/server/migrations`. SQLite and PostgreSQL share the same schema. The
history contains one root revision that creates the current schema for a fresh
database. Startup accepts only an empty unversioned database or the current
revision; it never infers or silently adopts an unknown schema. Databases stamped
with removed historical revisions fail closed and must be recreated. Startup
does not drop or recreate them. The current schema includes
the `application_settings` singleton and immutable `training_snapshots` and
`training_snapshot_rows` tables.

`DatabaseManager` is the stable owner of the active engine, session factory,
and transaction boundary. Repositories and services keep that manager rather
than retaining independent pools. The Database Settings endpoint verifies and
migrates a candidate before draining HTTP requests and session leases, then
adopts its engine and reloads the candidate database's settings. Active jobs
and public-data sync runs block switching; failed activation restores the
previous manager and configuration.

Database connection settings are stored in `database.json` under the
application data root, independent of either backend. The versioned file is
validated before the database is opened and replaced atomically. Passwords
use Windows DPAPI on Windows; elsewhere the file is restricted to mode 0600.
Application settings and operational records have one authority: the active
database. Switching databases never copies settings or operational data. A
missing `database.json` selects the current default SQLite configuration;
successful fresh initialization seeds typed settings before persisting the file.
Retired environment variables and older JSON settings files are not imported.

SQLite keeps foreign-key enforcement enabled during schema initialization and
must pass `PRAGMA foreign_key_check` before commit.

SQLite transactions serialize writes within the single backend process to
avoid lock-upgrade collisions when independent public-data runs are queued or
updated concurrently. PostgreSQL continues to use database-managed transaction
concurrency.

## ML artifacts

ML does not open the operational database. It receives snapshots through the
in-process `TrainingDataAccess` contract, checks the content hash, and keeps its
training manifest and checkpoints under the configured storage root. This
prevents ORM changes from becoming an implicit ML API.
