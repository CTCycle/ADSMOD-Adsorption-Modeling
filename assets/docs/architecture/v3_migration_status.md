# Canonical architecture status

Last updated: 2026-10-21

The architecture cutover is complete for the current runtime:

- one validated `UserSettings` record stored in the active database and edited
  through the Settings page, assembled with connection values from the
  independent data-root `database.json`;
- one Database Settings tab for SQLite/PostgreSQL connection testing and safe
  runtime switching, with no automatic cross-database data copying;
- one Hatch backend package and lockfile;
- repository-owned migrations, operational persistence, and training snapshots;
- ML isolated behind a lazy optional dependency profile and in-process snapshot
  access;
- versioned `/api/v1` Core and ML routes with explicit health endpoints;
- Angular configuration and polling driven by service responses;
- launcher, tests, CI, editor settings, scripts, and OpenAPI snapshots aligned
  with the canonical layout.

Validation is intentionally split into local unit/integration checks and live
service/browser checks. A passing static or schema check does not claim that a
database provider, browser session, or long-running training process was
available.
