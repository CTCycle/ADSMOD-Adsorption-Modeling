# ADSMOD runtime modes

Last updated: 2026-08-30

## `core`

The launcher starts Core and the Angular client. Core owns health, capabilities,
dataset import, NIST ingestion, fitting, database initialization, and the
internal snapshot API. Training is reported as not configured.

## `core-ml`

The launcher starts Core, ML, and the Angular client. ML exposes the training
API, authenticates to Core, retrieves immutable snapshots, verifies their
hashes, and stores its manifest and checkpoints in the configured storage root.

## Test execution

`app/tests/run_tests.bat` reads the same canonical configuration, starts only
the services required by the selected mode, waits for readiness, and runs the
Python and frontend checks. ML E2E tests are skipped in `core` mode.

There is no unified backend mode and no alternate compatibility mode.
