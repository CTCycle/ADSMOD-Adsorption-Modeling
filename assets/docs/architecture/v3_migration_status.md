# ADSMOD v3 Migration Status

Last updated: 2026-07-11

## Canonical runtime boundary

The v3 core service is being introduced under `app/backend/core` as the `adsmod-core` package. It accepts one explicit JSON configuration through `--config` and exposes:

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/system/capabilities`

The shared contract package is `app/backend/common`, published as `adsmod-common`. It contains only version, configuration, health, capability, and error-envelope contracts.

Canonical configuration lives under `settings/adsmod.json` with its JSON schema in `settings/adsmod.schema.json`. The old `settings/.env`, `settings/core_service.json`, and `settings/ml_service.json` remain legacy inputs until the service extraction and launcher migration phases are complete; the new core runtime does not read them.

## Transitional boundary

The legacy `app/server` services remain active while routes, persistence, and jobs are migrated. The legacy combined ASGI entrypoint, shared backend environment, frontend proxy, and PowerShell launcher are not yet part of the v3 runtime.

Do not describe the repository as fully migrated until those legacy paths are removed and core-plus-ML integration is validated.
## Core-owned snapshots

Core now owns an immutable SQLite-backed snapshot store and exposes authenticated internal endpoints:

- `POST /api/v1/internal/snapshots`
- `GET /api/v1/internal/snapshots/{snapshot_id}`

Snapshots are serialized canonically, identified by UUID, and returned with a SHA-256 content hash and paginated rows. The ML service will consume this contract after its package extraction; it must not access the core database directly.

## ML package boundary

The independent pp/backend/ml package exposes ML health and capability contracts and consumes core-owned snapshots through CoreSnapshotClient. It has no imports from dsmod_core, core_service, or the shared legacy persistence package. Snapshot pages are hash-verified before training data preparation can consume them.
