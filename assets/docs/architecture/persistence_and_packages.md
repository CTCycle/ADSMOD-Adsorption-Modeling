# Persistence and packages

Last updated: 2026-11-09

## Backend package

`app/server/pyproject.toml` and `app/server/uv.lock` define one Hatch package
named `adsmod-backend-workspace`. The package is rooted at `app/server`, with
the `ml` extra selecting the optional learning dependencies. The launcher and
CI install this package and use its lockfile.

## Core persistence

Core owns the active database and its Alembic baseline under
`app/server/migrations`. SQLite and PostgreSQL share the same schema. The
history starts from one root revision (`20261021_current_schema`) that creates
the baseline schema; later revisions chain linearly from it (currently
`20261106_modeling_sessions`; earlier steps add `custom_units` and dataset editing). Startup upgrades
an empty or baseline-stamped database to the head and accepts only that head; it never infers or silently adopts an unknown schema. Databases stamped
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

## Fitting sessions

`fitting_sessions` (one per dataset-level run, with a `configuration` JSON
snapshot of effective bounds, overrides, and experiment input hashes) owns
`fitting_session_models`, `fit_results` (`session_id`; `isotherm_id` is SET NULL
and `isotherm_name` is kept so results survive experiment deletion), and
`fit_parameters`. `adsorption_model_settings` holds one row per model
(`model_key`, `enabled`, `parameter_bounds` JSON in canonical units,
`updated_at`) and supplies the global solver bounds; the former
`application_settings.fitting_default_parameter_*` columns were dropped.
Migration `20261106_modeling_sessions` drops the old `fitting_runs` tables, so
previously stored single-experiment runs are intentionally discarded. The
dataset delete-impact field is `fitting_session_count`.

## ML artifacts

ML does not open the operational database. It receives snapshots through the
in-process `TrainingDataAccess` contract, checks the content hash, and keeps its
training manifest and checkpoints under the configured storage root. This
prevents ORM changes from becoming an implicit ML API.

## Adsorption data hierarchy

Adsorption data is stored as dataset → experiment → measurement:

- `datasets` is a collection of one or more experiments (`created_at`, and
  `updated_at`, exposed as `last_update`, which is bumped on every saved edit
  and whenever experiments are appended).
- `isotherms` is the experiment table: one dataset, exactly one adsorbent
  (`adsorbent_id`, RESTRICT), one temperature, pressure basis, and optional
  saturation pressure. `edited_at` records the last local edit; NIST
  re-download restores edited experiments only when explicitly confirmed.
- `isotherm_components` links an experiment to one or more adsorbates
  (RESTRICT); single-species isotherms have one component, NIST binary
  mixtures and uploaded multi-species experiments one per species. Upload
  files with one uptake column per species are melted to this long form
  during import, so no per-species column is ever stored.
- `observations` holds measurements with source values and units next to the
  canonical pressure (Pa or p/p₀) and uptake (mol/kg).

`adsorbents` and `adsorbates` are reusable across datasets; deleting a dataset
cascades to its experiments, measurements, and fitting sessions but never removes
materials or species, and a material or species cannot be deleted while an
experiment references it. Edits overwrite stored values in one transaction
(`DatasetEditRepository.apply`); there is no version history. Edited source
values are re-converted with the importer's `UnitCatalog`, so canonical values
follow the same unit rules as imports.

