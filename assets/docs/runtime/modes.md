# ADSMOD installation profiles

Last updated: 2026-11-10

ADSMOD has one runtime architecture and one FastAPI backend process. Optional
deep learning support is a dependency set managed from Settings, not a separate
runtime service.

## Lightweight (default) installation

The launcher installs the core backend dependencies and Angular client only. The
backend provides health, capabilities, dataset import, public data access,
fitting, database initialization, and all non-ML application functions.
`/api/v1/system/capabilities` reports machine learning as unavailable, and the
frontend does not expose training navigation or routes.

## Deep learning installed from Settings

**Settings > Deep learning** installs the `ml` dependency extra into the
existing virtual environment (see `operations/troubleshooting.md` for states
and recovery). Training routes are always registered but gated per request on
the verified status; the ML stack is imported lazily on the first training
call. Uninstall removes only packages exclusive to the extra, computed from
`uv.lock`. Training data is obtained
through the shared in-process contract, while checkpoints and other artifacts
use the configured storage root.

## Test execution

Automated validation covers both states. Gating tests simulate missing and
broken packages; tests that need the real stack wait for the background
verification and assume the development environment has it installed.

The bounded live ML lifecycle is validated separately from these deterministic
automated gates. It does not establish representative-data quality, convergence,
long-duration stability, or release readiness.
