# ADSMOD architecture

Last updated: 2026-08-30

ADSMOD has a single source of truth for runtime configuration and a strict
service split:

- `app/client` owns the Angular user interface and uses same-origin versioned
  API paths.
- `app/backend/common` contains framework-neutral contracts, configuration,
  health, paths, and version data.
- `app/backend/core` owns the operational database, Alembic migrations,
  dataset/NIST/fitting workflows, and immutable training snapshots.
- `app/backend/ml` owns model execution, training artifacts, and checkpoints.
  It retrieves authenticated snapshots from Core over HTTP and never imports
  the ORM or migration layer.

The launcher, CI, scripts, editor configuration, tests, and documentation all
target this layout. There is no second runtime or compatibility route surface.
