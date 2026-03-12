# ADSMOD Packaging and Runtime Modes

## 1. Strategy

ADSMOD uses one active runtime file: `ADSMOD/settings/.env`.

- Supported runtime paths are local-only:
  - Local webapp mode via `ADSMOD/start_on_windows.bat`
  - Packaged desktop mode via Windows Tauri artifacts
- Mode switching is configuration-only: replace values in `ADSMOD/settings/.env`.
- Only local deployment paths are supported.

## 2. Runtime Profiles

- `ADSMOD/settings/.env.local.example`: local webapp defaults (loopback host values, embedded DB).
- `ADSMOD/settings/.env.local.tauri.example`: desktop packaging/runtime defaults.
- `ADSMOD/settings/.env`: active profile used by launcher, tests, and packaged runtime startup.
- `ADSMOD/settings/configurations.json`: non-runtime defaults only (no database runtime settings).

## 3. Required Environment Keys

| Key | Purpose |
|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind host/port. |
| `UI_HOST`, `UI_PORT` | Frontend host/port used by local webapp launcher mode. |
| `VITE_API_BASE_URL` | Frontend API base path. Must stay `/api` for same-origin proxying. |
| `RELOAD` | Enables backend reload in local development when `true`. |
| `OPTIONAL_DEPENDENCIES` | Enables optional test dependencies in local launcher flow. |
| `DB_EMBEDDED` | `true` uses SQLite; `false` uses external DB settings. |
| `DB_ENGINE`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | External DB connection settings used when `DB_EMBEDDED=false`. |
| `DB_SSL`, `DB_SSL_CA` | External DB TLS settings. |
| `DB_CONNECT_TIMEOUT`, `DB_INSERT_BATCH_SIZE` | DB connection and write-batching runtime settings. |
| `MPLBACKEND`, `KERAS_BACKEND` | Runtime backend settings for plotting and ML stack. |

## 4. Local Webapp Mode (Default)

1. Copy local profile values into active env:
   - `copy /Y ADSMOD\settings\.env.local.example ADSMOD\settings\.env`
2. Start application:
   - `ADSMOD\start_on_windows.bat`
3. Run tests (optional):
   - `tests\run_tests.bat`

## 5. Packaged Desktop Mode (Windows Tauri)

1. Copy desktop profile values into active env:
   - `copy /Y ADSMOD\settings\.env.local.tauri.example ADSMOD\settings\.env`
2. Ensure portable runtimes are present at least once:
   - `ADSMOD\start_on_windows.bat`
3. Build desktop artifacts:
   - `release\tauri\build_with_tauri.bat`

Rust packaging prerequisite:
- `cargo` available in `PATH`.
- default toolchain configured (recommended: `stable-x86_64-pc-windows-msvc`).

Exported artifacts are generated in:
- `release/windows/installers`
- `release/windows/portable`

## 6. Deterministic Build Notes

- Backend dependency graph is lockfile-backed via `runtimes/uv.lock` (staged as `uv.lock` during sync/bundle) and installed with `uv sync --frozen`.
- Frontend dependency graph is lockfile-backed via `ADSMOD/client/package-lock.json` and installed with `npm ci`.
- Desktop packaging pipeline is implemented under `release/tauri/`.

