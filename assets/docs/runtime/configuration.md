# ADSMOD Runtime Configuration

Last updated: 2026-06-03

## Primary Runtime Files

- Environment file: `ADSMOD/settings/.env`
- Structured settings file: `ADSMOD/settings/configurations.json`

## Environment Variables

Current launcher and runtime keys include:

- `CORE_SERVICE_HOST`
- `CORE_SERVICE_PORT`
- `CORE_SERVICE_RELOAD`
- `ML_SERVICE_HOST`
- `ML_SERVICE_PORT`
- `ML_SERVICE_RELOAD`
- `FASTAPI_HOST`
- `FASTAPI_PORT`
- `UI_HOST`
- `UI_PORT`
- `OPTIONAL_DEPENDENCIES`
- `MPLBACKEND`
- `KERAS_BACKEND`
- `VITE_API_BASE_URL`

## Structured Settings Coverage

`ADSMOD/settings/configurations.json` contains:

- database mode and connection settings
- job polling interval
- dataset, NIST, fitting, and training defaults

## Mode-Specific Configuration Behavior

- Local launcher mode
  - uses `.env` host and port values
  - unified backend startup uses `CORE_SERVICE_*` values with fallback to `FASTAPI_*`
  - runs backend and frontend as separate processes
- API-only mode
  - requires no frontend process
  - is best suited for backend debugging
- Tauri mode
  - sets `ADSMOD_TAURI_MODE=true`
  - serves packaged frontend assets inside the desktop shell
- Test mode
  - reads `.env`
  - normalizes wildcard hosts such as `0.0.0.0` and `::` to `127.0.0.1` for client access
