# ADSMOD System Overview

Last updated: 2026-08-16

## Platform Shape

ADSMOD is a Windows-first local application with:

- Canonical v3 backend packages under `app/backend`
  - `common` for versioned configuration, health, capability, and error contracts.
  - `core` for the core ASGI application, CLI, and immutable snapshot store.
  - `ml` for the independent ML health/capability service and core snapshot client.
- Active backend services under `app/server`
  - `core_service` for non-ML API workflows such as health, datasets, fitting, and NIST.
  - `ml_service` for training datasets, checkpoints, and training lifecycle workflows.
  - `shared` for persistence, repositories, schemas, and common backend utilities.
- One frontend
  - `app/client` for datasets, dashboards, fitting, and training workflows.
  - Training remains visible in the unified UI but depends on the optional ML service at runtime.
- Runtime bootstrap assets under `runtimes/`.

Canonical v3 configuration is `app/resources/adsmod.json`, validated by
`app/resources/adsmod.schema.json`. Operational environment toggles remain in
`settings/.env.example`; service-specific JSON configuration is not supported.

## Backend Package Layout

```text
app/server/
  pyproject.toml
  uv.lock

  core_service/
    pyproject.toml
    core_service/
      app.py
      api/
      configurations/
      domain/
      services/
        data/
          import_parser.py
          nist_mapper.py

  ml_service/
    pyproject.toml
    ml_service/
      app.py
      api/
      domain/
      services/
      learning/

  shared/
    pyproject.toml
    shared/
      common/
      models/
      repositories/
        database/
        queries/       # training queries only
        schemas/
        datasets.py
        fitting.py
        materials.py
        nist.py        # sole NIST persistence/query owner
      services/

app/backend/
  common/       # adsmod-common
  core/         # adsmod-core
  ml/           # adsmod-ml
```

## Service Entry Points

- Core ASGI app: `core_service.app:app`
- ML ASGI app: `ml_service.app:app`
- Unified local-web backend composition entrypoint: `app.server.app:app`

v3 entrypoints are created through `adsmod_core.create_app_from_path(...)` and
`adsmod_ml.create_app(...)`; the core CLI accepts an explicit `--config` path.

The unified entrypoint composes core routes and, when explicitly enabled, ML
routes; it does not own backend business handlers. Core-only construction keeps
ML packages out of the import graph through the documented lazy ML composition
loader.

NIST provider frames are mapped by
`core_service.services.data.nist_mapper.NISTCanonicalMapper`. Canonical NIST
reads and writes are owned by `shared.repositories.nist.NISTRepository`.

## Frontend Responsibility

- `app/client` routes `datasets`, `public-data`, `public-materials`, `dashboards`, `fitting`, and `training`.
- `datasets` is limited to user-uploaded custom datasets.
- `public-data` owns NIST adsorption experiments; `public-materials` owns NIST adsorbates and adsorbent materials plus the existing PubChem enrichment action.
- `/api/training/*` traffic is routed to `ml_service` in development proxy mode.
- Other `/api/*` traffic is routed to `core_service`.
- In core-only mode, training routes show an unavailable state instead of failing the Custom Datasets, Public Data, Public Materials, or Fitting workflows.
