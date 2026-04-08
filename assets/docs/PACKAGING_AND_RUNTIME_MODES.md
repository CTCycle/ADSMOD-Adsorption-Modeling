# ADSMOD Packaging and Runtime Modes

Last updated: 2026-04-08

## 1. Runtime Strategy

Single active runtime profile:
- `ADSMOD/settings/.env`

Supported modes:
- local webapp mode (launcher flow),
- packaged desktop mode (Windows Tauri artifacts).

Switching mode is configuration-driven by editing the single `.env` runtime file.

## 2. Runtime Profiles

- `ADSMOD/settings/.env`: active runtime file used by launcher/tests/runtime startup.
- `ADSMOD/settings/configurations.json`: application defaults, including database mode/connection settings.

## 3. Local Runtime Assets

- Python runtime: `runtimes/python`.
- Virtual environment: `runtimes/.venv`.
- uv runtime/cache/lock: `runtimes/uv`, `runtimes/.uv-cache`, `runtimes/uv.lock`.
- Node runtime: `runtimes/nodejs`.

## 4. Common Environment Keys

| Key | Purpose |
|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind host/port. |
| `UI_HOST`, `UI_PORT` | Frontend host/port in launcher and test flows. |
| `VITE_API_BASE_URL` | Frontend API base path (`/api` expected for same-origin mode). |
| `RELOAD` | Backend reload toggle in local development. |
| `OPTIONAL_DEPENDENCIES` | Enables optional test dependencies in launcher flow. |
| `MPLBACKEND`, `KERAS_BACKEND` | Scientific/ML backend runtime behavior. |

Database settings are loaded from `ADSMOD/settings/configurations.json` (`database` section), not from `.env`.

## 5. Local Webapp Mode

1. Configure runtime/process keys in `ADSMOD/settings/.env`.
2. Bootstrap and run:
   - `ADSMOD\start_on_windows.bat`
3. Optional tests:
   - `tests\run_tests.bat`

## 6. Desktop Packaging Mode (Windows Tauri)

1. Configure runtime/process keys in `ADSMOD/settings/.env`.
2. Ensure runtimes/deps are prepared at least once:
   - `ADSMOD\start_on_windows.bat`
3. Build/export desktop artifacts:
   - `release\tauri\build_with_tauri.bat`

Prerequisite:
- Rust `cargo` on `PATH` with a configured toolchain.

Expected output directories:
- `release/windows/installers`
- `release/windows/portable`

## 7. Determinism Notes

- Python dependencies are synchronized through `uv` using lockfile flow (`runtimes/uv.lock` staged to project `uv.lock` during launcher sync).
- Frontend dependencies are lockfile-backed (`ADSMOD/client/package-lock.json`) with `npm ci` when lockfile exists.
