# Canonical architecture status

Last updated: 2026-08-30

The architecture cutover is complete for the current runtime:

- one validated `AdsmodConfig` loaded from `app/resources/adsmod.json`;
- one backend workspace and lockfile;
- Core-owned migrations, operational persistence, and training snapshots;
- ML isolated behind authenticated snapshot requests;
- versioned `/api/v1` Core and ML routes with explicit health endpoints;
- Angular configuration and polling driven by service responses;
- launcher, tests, CI, editor settings, scripts, and OpenAPI snapshots aligned
  with the canonical layout.

Validation is intentionally split into local unit/integration checks and live
service/browser checks. A passing static or schema check does not claim that a
database provider, browser session, or long-running training process was
available.
