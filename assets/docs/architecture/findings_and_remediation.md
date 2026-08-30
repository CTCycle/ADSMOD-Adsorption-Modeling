# Findings and remediation

Last updated: 2026-08-30

The original repository contained a unified legacy web runtime alongside an
unfinished split backend. That older runtime combined Core, ML, and persistence
through import paths and environment switches, which made it unclear which
implementation was active. It has no current runtime purpose and is retired.

The remediation is now complete:

1. Core and ML were moved behind explicit package boundaries.
2. Core became the only owner of SQLAlchemy, Alembic, and the operational
   database.
3. ML now consumes authenticated, hash-verified Core snapshots.
4. The Angular client, launcher, tests, CI, editor configuration, scripts, and
   documentation were changed to the same config and route contracts.
5. Legacy schema adoption/fingerprint behavior and compatibility aliases were
   removed rather than preserved.

Remaining verification is operational rather than architectural: run the
launcher on the target Windows host, exercise the browser workflow, and run
the optional `core-ml` path when its model dependencies and service token are
available.
