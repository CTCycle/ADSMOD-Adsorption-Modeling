# ADSMOD troubleshooting

Last updated: 2026-09-21

## Backend or UI unreachable

- Check the backend and frontend host/port values in `app/resources/adsmod.json`.
- Confirm the backend responds at `/health/ready`.
- Confirm `app/client/dist/browser/index.html` exists and the frontend preview
  serves the built Angular application.
- Check backend and frontend launcher logs under the configured storage root.

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

Use **Stop application** for processes started by the current launcher session.
A process left by an earlier session can be handled with **Kill all
application processes** when its command line is recognized as an ADSMOD
backend or frontend process. Unrelated processes remain untouched.

## Missing dependencies

Run **Install / update dependencies** in `start_on_windows.ps1`. The installer
uses the locked backend workspace and `npm ci`, and writes the dependency-state
manifest only after the installation succeeds. Automatic repair preserves the
recorded Standard/Development and Base/ML profile; choose the ML-enabled
install option when training functionality is required. Resolve filesystem or
network errors and rerun the action rather than reusing a partial environment.

If a state manifest is missing or stale, the first normal launch may perform
one synchronization. A stale frontend build state causes a build after
dependency repair; a current dependency/build state skips both operations.

## Training unavailable

Query `/api/v1/system/capabilities`. If `features.machine_learning` is false,
the backend is running correctly without the optional ML extension. Re-run the
installer with machine learning dependencies enabled, then relaunch ADSMOD.
The frontend intentionally hides or blocks training routes while that
capability is unavailable.

If ML dependencies are installed but the capability remains false, inspect the
backend startup log for the recorded optional-extension load reason and verify
that the local Python and compute environment can import the installed ML
stack.

## Database startup failure

Inspect the backend logs below the configured storage root. Unknown or
non-empty unversioned schemas are rejected deliberately; export or repair them
manually instead of stamping a guessed revision. For PostgreSQL, verify
connectivity, credentials, and the role's database-creation permission.
