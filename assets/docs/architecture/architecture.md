# Architecture

Last updated: 2026-08-20

Canonical architecture details live in:

- [`architecture/overview.md`](overview.md)
- [`architecture/system_overview.md`](system_overview.md)
- [`architecture/service_boundaries.md`](service_boundaries.md)
- [`architecture/api_surface.md`](api_surface.md)
- [`architecture/persistence_and_packages.md`](persistence_and_packages.md)
- [`architecture/v3_migration_status.md`](v3_migration_status.md)
- [`architecture/findings_and_remediation.md`](findings_and_remediation.md)

The repository currently contains two backend generations: the v3 package
boundaries under `app/backend` and the launcher-selected active service
workspace under `app/server`. The active workspace is not a compatibility
layer; it is the current public runtime until each vertical slice is replaced
atomically.
