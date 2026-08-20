# Architecture Overview

Last updated: 2026-08-20

## Scope

This section covers system structure, ownership boundaries, API routing, and
backend workspace organization.

The repository has two explicitly separated backend generations:

- `app/server` is the launcher-selected active local-web runtime.
- `app/backend` contains the extracted v3 package boundaries and snapshot
  contract, but is not yet the launcher-selected runtime.

The active runtime remains the current public contract. Internal replacements
are atomic: update every caller, then delete the superseded implementation.
There are no compatibility aliases, fallback configuration paths, or duplicate
internal authorities.

## Documents

- [`system_overview.md`](system_overview.md)
  - Current runtime, repository layout, entry points, and source-of-truth rules.
- [`service_boundaries.md`](service_boundaries.md)
  - Current and target dependency direction, ownership, and prohibited imports.
- [`api_surface.md`](api_surface.md)
  - Core/ML route ownership and critical request/training flows.
- [`persistence_and_packages.md`](persistence_and_packages.md)
  - Package ownership, ORM class map, twelve-table ER diagram, and v3 snapshots.
- [`v3_migration_status.md`](v3_migration_status.md)
  - Extracted package status and snapshot migration boundary.
- [`findings_and_remediation.md`](findings_and_remediation.md)
  - Repository-wide findings, priorities, target state, and remediation sequence.

## When to read which file

- Read [`system_overview.md`](system_overview.md) for orientation and directory layout.
- Read [`service_boundaries.md`](service_boundaries.md) before moving backend code or changing imports.
- Read [`api_surface.md`](api_surface.md) before adding or relocating endpoints.
- Read [`persistence_and_packages.md`](persistence_and_packages.md) before changing repositories, models, sessions, or packaging.
- Read [`v3_migration_status.md`](v3_migration_status.md) before changing v3 packages, snapshots, or the configuration boundary.
- Read [`findings_and_remediation.md`](findings_and_remediation.md) before planning cross-cutting architectural work.
