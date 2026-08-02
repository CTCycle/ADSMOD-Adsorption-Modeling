# ADSMOD v3 Migration Status

Last updated: 2026-08-02

## Canonical runtime boundary

The v3 core service is implemented under `app/backend/core` as the `adsmod-core`
package. It accepts one explicit JSON configuration through `--config` and exposes:

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/system/capabilities`

The shared contract package is `app/backend/common`, published as `adsmod-common`. It contains only version, configuration, health, capability, and error-envelope contracts.

Canonical configuration lives under `app/resources/adsmod.json` with its JSON schema in `app/resources/adsmod.schema.json`. The extracted v3 packages, transitional backend, launcher, proxy, and tests read this resource; no older service configuration files or path aliases remain.

## Runtime boundary

The unified backend remains the application route owner, while the extracted core
and ML packages own their versioned contracts. They share the canonical resource
without compatibility loaders, redirects, or fallback configuration paths.
## Core-owned snapshots

Core now owns an immutable SQLite-backed snapshot store and exposes authenticated internal endpoints:

- `POST /api/v1/internal/snapshots`
- `GET /api/v1/internal/snapshots/{snapshot_id}`

Snapshots are serialized canonically, identified by UUID, and returned with a SHA-256 content hash and paginated rows. The extracted ML package provides the client boundary for this contract; the current transitional ML workflow must not access the core database directly.

## ML package boundary

The independent `app/backend/ml` package exposes ML health and capability contracts
and consumes core-owned snapshots through `CoreSnapshotClient`. It has no imports
from `adsmod_core`, `core_service`, or the shared legacy persistence package.
The extracted ML package includes a `CoreSnapshotClient` that paginates and
hash-verifies snapshot pages. Full v3 training-data consumption and launcher
integration remain pending; the current training UI still uses the transitional
ML service.
