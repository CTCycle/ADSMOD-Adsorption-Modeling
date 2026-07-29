# ADSMOD Runtime Configuration

Last updated: 2026-07-29

## Canonical v3 Configuration

- Configuration file: `app/resources/adsmod.json`
- Validation schema: `app/resources/adsmod.schema.json`
- Required sections: `version`, `runtime`, `storage`, `security`, and `application`
- Supported runtime modes: `core` and `core-ml`
- The v3 core CLI receives the configuration explicitly through `--config`.

All backend packages and the launcher read this file. There are no service-specific
configuration files or configuration-path aliases.

## Environment Variables

Operational environment keys are limited to:

- `BACKEND_LOGS_VISIBLE`
- `ALWAYS_REBUILD`
- `RELOAD`
- `MPLBACKEND`
- `KERAS_BACKEND`
- `VITE_API_BASE_URL`

## Structured Settings Coverage

The canonical `application` section contains:

- database settings
- dataset, NIST, fitting, job, and training defaults

The `runtime` section is the only source for backend, ML, and frontend hosts and ports.

## Mode-Specific Configuration Behavior

- Local launcher mode
  - uses `app/resources/adsmod.json` for hosts, ports, and application defaults
  - shows backend logs in a separate terminal when `BACKEND_LOGS_VISIBLE=true`; defaults to visible when absent
  - rebuilds the frontend at application start when `ALWAYS_REBUILD=true`; defaults to rebuilding when absent
  - reads only the canonical configuration resource
  - runs backend and frontend as separate processes
- API-only mode
  - requires no frontend process
  - is best suited for backend debugging
- Test mode
  - reads the canonical runtime resource
  - normalizes wildcard hosts such as `0.0.0.0` and `::` to `127.0.0.1` for client access
