# ADSMOD v3 Migration Status

Last updated: 2026-08-20

## Canonical package boundary

The v3 core service is implemented under `app/backend/core` as the
`adsmod-core` package. It accepts one explicit JSON configuration through
`--config` and exposes:

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/system/capabilities`

The shared package is `app/backend/common`, published as `adsmod-common`. It
contains the canonical `AdsmodConfig` shape plus health, capability, error, and
other v3 contracts. `app/backend/core` and `app/backend/ml` depend on it and do
not import transitional service packages.

Canonical runtime values live in `app/resources/adsmod.json`; the checked-in
`app/resources/adsmod.schema.json` is generated from `AdsmodConfig`. The
extracted packages, active backend, launcher, proxy, and tests use that one
resource (or the explicit `ADSMOD_RESOURCES_DIR` root). No older service files,
path aliases, compatibility loaders, or fallback locations remain.

## Runtime boundary

The launcher still starts the unified `app.server.app` application. The
extracted core and ML packages own their versioned contracts and are tested
independently; they are not imported as aliases for the active services. The
active runtime owns the current public HTTP paths and payloads while the v3
cutover is staged by vertical slice.

## Core-owned snapshots

Core owns an immutable SQLite-backed snapshot store and exposes authenticated
internal endpoints:

- `POST /api/v1/internal/snapshots`
- `GET /api/v1/internal/snapshots/{snapshot_id}`

Snapshots are serialized canonically, identified by UUID, and returned with a
SHA-256 content hash and paginated rows. The extracted ML package provides the
client boundary for this contract.

## ML package boundary

The independent `app/backend/ml` package exposes ML health and capability
contracts and consumes core-owned snapshots through `CoreSnapshotClient`. It
has no imports from `adsmod_core`, `core_service`, or the transitional shared
persistence package. Full v3 training-data consumption and launcher
integration remain pending; the current training UI still uses the active ML
service and shared operational database.

At cutover, callers move to the snapshot contract and the replaced direct
database reader is deleted in the same change. No dual readers or fallback
behavior is planned.
