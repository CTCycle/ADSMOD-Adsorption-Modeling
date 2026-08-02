# Architecture

Last updated: 2026-08-02

Canonical architecture details live in:

- [`architecture/overview.md`](overview.md)
- [`architecture/system_overview.md`](system_overview.md)
- [`architecture/service_boundaries.md`](service_boundaries.md)
- [`architecture/api_surface.md`](api_surface.md)
- [`architecture/persistence_and_packages.md`](persistence_and_packages.md)
- [`architecture/v3_migration_status.md`](v3_migration_status.md)
  - Canonical v3 packages, internal snapshot contract, and migration boundary.

The repository currently contains two backend generations: the canonical v3 packages
under `app/backend` and the transitional service workspace under `app/server`.
