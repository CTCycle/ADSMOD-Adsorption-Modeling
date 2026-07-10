# ADSMOD v3 Migration Status

Last updated: 2026-07-10

## Canonical runtime boundary

The v3 core service is being introduced under `app/backend/core` as the `adsmod-core` package. It accepts one explicit JSON configuration through `--config` and exposes:

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/system/capabilities`

The shared contract package is `app/backend/common`, published as `adsmod-common`. It contains only version, configuration, health, capability, and error-envelope contracts.

Canonical configuration lives under `settings/adsmod.json` with its JSON schema in `settings/adsmod.schema.json`. The old `settings/.env`, `settings/core_service.json`, and `settings/ml_service.json` remain legacy inputs until the service extraction and launcher migration phases are complete; the new core runtime does not read them.

## Transitional boundary

The legacy `app/server` services remain active while routes, persistence, and jobs are migrated. The legacy combined ASGI entrypoint, shared backend environment, frontend proxy, launchers, and Tauri process manager are not yet part of the v3 runtime.

Do not describe the repository as fully migrated until those legacy paths are removed and core-plus-ML integration is validated.