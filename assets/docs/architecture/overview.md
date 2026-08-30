# Architecture documentation

Last updated: 2026-08-30

The canonical runtime consists of one Angular client and two explicitly
separated Python services. This folder documents ownership and boundaries; the
checked-in source and `app/resources/adsmod.json` remain authoritative.

- [`system_overview.md`](system_overview.md): repository layout and runtime flow.
- [`service_boundaries.md`](service_boundaries.md): dependency direction and
  import restrictions.
- [`api_surface.md`](api_surface.md): public, internal, and health endpoints.
- [`persistence_and_packages.md`](persistence_and_packages.md): package and
  database ownership.
- [`v3_migration_status.md`](v3_migration_status.md): completed cutover status.
- [`findings_and_remediation.md`](findings_and_remediation.md): resolved
  architecture findings and remaining operational checks.
