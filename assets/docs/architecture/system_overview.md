# System overview

Last updated: 2026-08-30

## Repository layout

```text
app/
  backend/
    common/    adsmod_common: contracts and canonical configuration
    core/      adsmod_core: HTTP API, persistence, migrations, snapshots
    ml/        adsmod_ml: training API, model runtime, snapshot client
    migrations/              Alembic package-local configuration
    pyproject.toml
    uv.lock
  client/      Angular application
  resources/   adsmod.json and generated configuration schema
  scripts/     maintenance and schema-generation entry points
  tests/       Python, browser, and integration validation
```

## Runtime flow

The Windows launcher reads `app/resources/adsmod.json`, synchronizes the
backend workspace, starts Core on `runtime.core_port`, optionally starts ML on
`runtime.ml_port` when `runtime.mode` is `core-ml`, and serves the Angular
bundle on `runtime.frontend_port`. Every process receives the same config path.

The browser reaches Core and ML through the development proxy. Core owns
database-backed data and publishes immutable snapshots for ML. ML authenticates
to Core, verifies the snapshot hash, and stores only its manifest and training
artifacts in the configured storage root.

## Configuration flow

`AdsmodConfig` validates the complete JSON document. Hosts, ports, storage,
database settings, fitting defaults, training defaults, and polling intervals
are not read from ad-hoc environment variables or frontend build-time files.
