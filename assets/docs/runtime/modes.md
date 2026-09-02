# ADSMOD installation profiles

Last updated: 2026-09-02

ADSMOD has one runtime architecture and one FastAPI backend process. Optional
machine learning support is an installation profile, not a separate runtime
service.

## Base installation

The launcher installs the core backend dependencies and Angular client. The
backend provides health, capabilities, dataset import, public data access,
fitting, database initialization, and all non-ML application functions.
`/api/v1/system/capabilities` reports machine learning as unavailable, and the
frontend does not expose training navigation or routes.

## ML-enabled installation

The launcher installs the backend with the `ml` dependency extra. The same
FastAPI application loads the ML extension in-process and registers training
configuration, jobs, checkpoints, and related routes. Training data is obtained
through the shared in-process contract, while checkpoints and other artifacts
use the configured storage root.

## Test execution

Automated validation covers both installation profiles independently. Base
validation asserts that the ML package is absent and the normal application is
ready. ML-enabled validation asserts that capability discovery enables machine
learning and that training endpoints are registered.

Live browser and hardware-specific training checks are intentionally separate
from these deterministic automated gates.
