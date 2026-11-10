# ADSMOD troubleshooting

Last updated: 2026-11-10

## Backend or UI unreachable

- Check `ADSMOD_HOST`, `ADSMOD_BACKEND_PORT`, and `ADSMOD_FRONTEND_PORT` in
  `settings/.env` or the process environment. Environment values override
  `.env`; restart ADSMOD after changing them.
- Check `settings/.env` for runtime CORS values and `ADSMOD_STORAGE_ROOT`.
  Database selection and connection values are in `database.json` under that
  data root.
- Confirm the backend responds at `/health/ready`.
- Confirm `app/client/dist/browser/index.html` exists; the backend serves it at
  its own URL, and without it `/` redirects to `/docs`.
- Check the launcher console and backend logs under the configured storage root.

The launcher uses dependency-state manifests and a content fingerprint for the
frontend build. Source, Angular configuration, package manifests, public
assets, the build verification script, or the pinned Node version can trigger
the corresponding repair/build. Documentation and test-only changes do not.
**Rebuild frontend** always performs an intentional frontend build.

## Configured port is already in use

The launcher checks all configured ports together before runtime or dependency
work. It reports every port, PID, and process name, deduplicates a process
that owns more than one port, and asks once whether to terminate the listed
processes.

- Answer `N` or run from a non-interactive console: startup is cancelled and
  no process is terminated.
- If an owner cannot be resolved, startup fails closed. Stop the owning
  application manually and retry.
- If ownership changes, termination is denied, a process fails to exit, or a
  port remains occupied after termination, startup is cancelled and the final
  owner is reported.
- A process that exits during the confirmation/termination race is treated as
  already resolved.

While ADSMOD runs, the launcher console shows the backend log; press Ctrl+C there
to stop it. If the launcher was force-killed and the backend survived, start the
launcher again and use **Kill all application processes**. Unrelated processes
remain untouched.

## Missing dependencies

Run **Install / update dependencies** in `setup_and_maintenance.ps1`. The installer
uses the locked backend workspace and `npm ci`, and writes the dependency-state
manifest only after the installation succeeds. Automatic repair preserves the
recorded Standard/Development profile and keeps any deep learning packages
that Settings installed. The launcher does not install them; use
**Settings > Deep learning**. Resolve filesystem or
network errors and rerun the action rather than reusing a partial environment.

If a state manifest is missing or stale, the first normal launch may perform
one synchronization. A stale frontend build state causes a build after
dependency repair; a current dependency/build state skips both operations.

## Training unavailable

Open **Settings > Deep learning** (or query `/api/v1/deep-learning/status`).
`features.machine_learning` in `/api/v1/system/capabilities` is true only when
the status is `installed`, meaning the packages exist **and** an import probe
succeeded. Training endpoints return 503 otherwise and the UI hides training.

- `not_installed`: lightweight mode; use **Install**.
- `failed`: read the recorded error code and hint. Typical causes are `network`
  (retry), `disk_full` (free about 8 GB), `file_locked` (close programs using
  the environment), `validation` (import failed; use **Repair / reinstall**),
  and `interrupted` (the installer was killed; use **Repair / reinstall**).
  Packages removed or changed outside ADSMOD are detected within about 30 s.
- `incompatible`: only Windows x64 is supported.
- A GPU warning means PyTorch cannot use CUDA (the locked build needs a driver
  supporting CUDA 13). Training still works on the CPU. ADSMOD never changes
  drivers.

The installer log is `<storage root>\deep_learning\install.log`. If ADSMOD
does not come back after an operation, start it with `start_on_windows.bat`;
if the environment is damaged, run **Install / update dependencies** in the
launcher menu.

## Database startup failure

Inspect the backend logs below the configured storage root. Historical or
unknown revisions and non-empty unversioned schemas are rejected deliberately.
Create a fresh database for the current baseline; startup never drops or
recreates an existing database. For PostgreSQL, verify connectivity,
credentials, and the role's database-creation permission.
