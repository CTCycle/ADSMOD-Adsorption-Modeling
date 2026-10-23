# Settings and `.env` validation — 2026-10-21

Scope: HEAD `8e75f04`; isolated ADSMOD storage root `runtimes/cache/settings-validation-20261021-8e75f04`; loopback backend `127.0.0.1:6045` and preview `127.0.0.1:5173`. No application source or API changes were made.

## Automated checks

- Focused backend settings/runtime/bootstrap slice: 25 passed. The bootstrap and runtime-configuration cases are now consolidated in `app/tests/unit/test_runtime_contracts.py`; settings API cases remain in `app/tests/backend/test_settings_api.py`.
- Frontend `npm test` — 27 files, 88 tests passed.
- Frontend `npm run test:preview` — 4/4 passed with loopback permission.
- `npm run lint` — passed; `npm run build` — passed.
- The backend slice includes temporary `.env` parsing, process-environment precedence, invalid endpoint rejection, runtime endpoint sourcing, application settings, database switch isolation, active-work switch blocking, and activation rollback coverage.

## Rendered settings UI

- Exercised all 31 application settings through the four application tabs. After debounced saves, the settings API returned 31/31 expected values; browser reload showed the saved values. A backend stop/relaunch on the same isolated data root preserved all 31 values.
- Invalid public-data max-record count `0` and fitting ceiling `1200` below default iterations `1500` were rejected with the UI validation message; the persisted values remained unchanged.
- Reset through the UI restored all 31 fields to the API defaults. Because `sync_resume_on_startup` had been changed in the running process, the restart notice remained until the next launcher relaunch; it then cleared. Final API state matched defaults and had no pending restart fields.
- Settings API contained only the 31 application-setting leaves and restart metadata. Runtime endpoint fields were absent. Runtime endpoint defaults and `.env`/process precedence are covered by the runtime/bootstrap tests.
- The rendered Settings page was captured and visually inspected in the in-app Browser. The Browser tool returned screenshot bytes for in-session inspection but no file-export operation; a data-URL export attempt was blocked by the Browser URL policy (only HTTP/HTTPS is allowed), so no PNG capture was persisted under `assets/QA/`.

## Database UI and storage

- The configured isolated SQLite connection and a new `data/secondary.db` candidate both passed Test connection.
- Switching to the secondary store loaded defaults; switching back to `data/database.db` restored the primary store's modified settings, demonstrating separate settings stores.
- A deliberately invalid candidate (`data/database.db/child.db`) failed activation and displayed “The new database could not be activated. The previous database remains selected.” Backend tests also passed activation rollback and active-job/sync blocking checks.
- Final database API response reported connected SQLite at `data/database.db`. No credentialed PostgreSQL connection, migration, live switch, or settings-persistence check was performed. A listener existed on port 5432 but was not used; PostgreSQL settings persistence remains unverified.

## `.env` and launcher

- Read `settings/.env` without editing it. Its SHA-256 was `E457328ACF71A54041C6CE897AEDC1EA4A6DC56497B552E1E9EF96D0D6088A42` before and after validation. Effective runtime defaults were host `127.0.0.1`, backend `6045`, frontend `5173`, and two derived CORS origins. The isolated launcher received a process-level storage-root override.
- Launcher started the backend to `/health/ready` and the preview; the in-app Browser loaded `/settings`. `/api/v1/settings` through the preview proxy returned 200.
- A second launch detected both occupied ports. Declining termination cancelled that launch and left readiness and proxy checks healthy. Launcher stop cleared ports 6045 and 5173; relaunch restored readiness and the default settings; final stop left both application ports clear. The unrelated port-5432 listener was left untouched.
- No provider acquisition was requested. Auto-sync was set to a 720-hour interval during the isolated settings exercise and reset to its disabled default before cleanup.
- The isolated data root was removed after validation. The existing `.env` and operational data were preserved.
