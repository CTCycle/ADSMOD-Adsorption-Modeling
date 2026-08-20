# ADSMOD Runtime Configuration

Last updated: 2026-08-20

## Canonical v3 Configuration

- Runtime value file: `<resource directory>/adsmod.json` (`app/resources` by default)
- Generated validation schema: `<resource directory>/adsmod.schema.json`
- Configuration-shape authority: `adsmod_common.config.AdsmodConfig`
- Required sections: `version`, `runtime`, `storage`, `security`, and `application`
- Supported runtime modes: `core` and `core-ml`
- The v3 core CLI receives the configuration explicitly through `--config`.

All backend packages, the launcher, and the frontend proxy read this value file.
There are no service-specific configuration files, configuration-path aliases,
duplicate validators, or fallback locations. The selected resource directory
must contain both canonical configuration files.

## Environment Variables

Operational environment keys are limited to:

- `BACKEND_LOGS_VISIBLE` controls launcher behavior.
- `RELOAD`, `MPLBACKEND`, and `KERAS_BACKEND` control local process behavior.
- `VITE_API_BASE_URL` controls the generated frontend runtime API base path.
- `ADSMOD_RESOURCES_DIR` overrides the default `app/resources` directory for
  the canonical configuration, logs, templates, checkpoints, and embedded
  SQLite database. Relative paths are resolved from the repository root.

The environment file does not override runtime hosts or ports.

## Structured Settings Coverage

The canonical `application` section contains:

- database settings
- dataset, NIST, fitting, job, and training defaults

The `runtime` section is the only source for backend, ML, and frontend hosts and ports.
The active `shared.common.settings` module returns typed projections of these
models; it does not parse JSON or accept independent dictionaries.

## Mode-Specific Configuration Behavior

- Local launcher mode
  - uses `$ADSMOD_RESOURCES_DIR/adsmod.json` for hosts, ports, and application defaults
    (`app/resources` by default)
  - shows backend logs in a separate terminal when `BACKEND_LOGS_VISIBLE=true`; defaults to visible when absent
  - repairs missing or unusable dependencies and frontend build output during launch; menu option 2, **Install / update dependencies**, always rebuilds the frontend, while menu option 3, **Rebuild frontend**, performs that frontend work without syncing backend dependencies
  - reads only the canonical configuration resource
  - runs backend and frontend as separate processes
- API-only mode
  - requires no frontend process
  - is best suited for backend debugging
- Test mode
  - reads the canonical runtime resource
  - normalizes wildcard hosts such as `0.0.0.0` and `::` to `127.0.0.1` for client access
