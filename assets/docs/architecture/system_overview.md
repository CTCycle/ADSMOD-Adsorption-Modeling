# System overview

Last updated: 2026-09-02

## Repository layout

```text
app/
  server/
    common/    adsmod_common: contracts and canonical configuration
    core/      adsmod_core: unified FastAPI application and persistence
    ml/        adsmod_ml: optional training extension loaded in-process
    migrations/              Alembic package-local configuration
    openapi/backend.json      canonical generated API contract
    pyproject.toml
    uv.lock
  client/      Angular application
  resources/   adsmod.json and generated configuration schema
  scripts/     maintenance and schema-generation entry points
  tests/       Python, browser, and integration validation
```

## Runtime flow

The Windows launcher reads `app/resources/adsmod.json`, synchronizes the
backend workspace, starts one FastAPI backend on the configured backend port,
and serves the Angular bundle on the configured frontend port. The browser
uses the same backend for datasets, NIST data, fitting, capability discovery,
and optional machine learning operations.

The core package owns database-backed data and persistence. When the ML extra
is installed, the unified application loads the ML extension and provides it
with in-process training-data access. No second backend process, internal HTTP
service token, or service-to-service proxy is required.

## Optional machine learning

The base backend installation does not install `adsmod_ml` or its heavy ML
dependencies. The optional `ml` dependency extra installs them. At startup the
backend detects whether the extension can be loaded and reports the result via
`/api/v1/system/capabilities`. The Angular client uses that capability as its
single source of truth for ML navigation and route access.

## Configuration flow

`AdsmodConfig` validates the complete JSON document. Hosts, ports, storage,
database settings, fitting defaults, training defaults, and polling intervals
are read from the canonical configuration rather than from ad-hoc frontend or
service-specific overrides.
