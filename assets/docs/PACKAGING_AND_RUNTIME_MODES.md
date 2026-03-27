# ADSMOD Packaging and Runtime Modes

## 1. Runtime Strategy

Single active runtime profile:
- `ADSMOD/settings/.env`

Supported modes:
- local webapp mode (launcher flow),
- packaged desktop mode (Windows Tauri artifacts).

Switching mode is configuration-driven by choosing the `.env` profile values.

## 2. Runtime Profiles

- `ADSMOD/settings/.env.local.example`: local webapp defaults.
- `ADSMOD/settings/.env.local.tauri.example`: desktop packaging defaults.
- `ADSMOD/settings/.env`: active runtime file used by launcher/tests/runtime startup.
- `ADSMOD/settings/configurations.json`: application defaults (not a runtime profile switch).

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
| `DB_EMBEDDED` | SQLite when `true`, external DB when `false`. |
| `DB_ENGINE`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | External DB settings. |
| `DB_SSL`, `DB_SSL_CA`, `DB_CONNECT_TIMEOUT`, `DB_INSERT_BATCH_SIZE` | DB security/performance settings. |
| `MPLBACKEND`, `KERAS_BACKEND` | Scientific/ML backend runtime behavior. |

## 5. Local Webapp Mode

1. Apply local profile:
   - `copy /Y ADSMOD\settings\.env.local.example ADSMOD\settings\.env`
2. Bootstrap and run:
   - `ADSMOD\start_on_windows.bat`
3. Optional tests:
   - `tests\run_tests.bat`

## 6. Desktop Packaging Mode (Windows Tauri)

1. Apply tauri profile:
   - `copy /Y ADSMOD\settings\.env.local.tauri.example ADSMOD\settings\.env`
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

