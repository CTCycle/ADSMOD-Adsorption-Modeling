# ADSMOD architecture

Last updated: 2026-09-16

ADSMOD has a single source of truth for runtime configuration and one layered
FastAPI package:

- `app/client` owns the Angular user interface and uses same-origin versioned
  API paths.
- `app/server/api` exposes the versioned HTTP routers.
- `app/server/common`, `app/server/configurations`, and `app/server/domain`
  contain shared helpers, configuration, and transport-neutral contracts.
- `app/server/services` owns core workflows; `app/server/repositories` owns
  the operational database, Alembic migrations, and immutable snapshots.
- `app/server/models` plus lazy ML-owned services provide model execution,
  training artifacts, and checkpoints when the `ml` extra is installed. They
  consume snapshots through the in-process contract and never import the ORM
  or migration layer.

The launcher, CI, scripts, editor configuration, tests, and documentation all
target this layout. There is no second runtime, compatibility package, or
compatibility route surface.
