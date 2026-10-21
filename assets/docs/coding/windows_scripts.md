# ADSMOD Windows Script Rules

Last updated: 2026-10-21

## Operational Script Expectations

- Use `start_on_windows.ps1` as the canonical launcher and maintenance entry point.
- Keep PowerShell usage explicit and deterministic.
- Preserve runtime-safe path handling.
- Keep disposable application and test-tool caches below `runtimes/cache`.
- Avoid destructive operations outside intended directories.
- Keep script behavior compatible with the existing launcher and runtime flow.
- Load runtime endpoints through `server.runtime_config` and connection
  settings from `database.json` below the configured storage root. Application
  settings remain in the selected database; `.env` may set only the storage
  root for database bootstrap.
- Check dependency readiness before reinstalling runtimes or packages.

## Documentation Expectation

- When documenting operational scripts, provide CMD and PowerShell forms where they differ.
- Prefer existing repository scripts over ad-hoc manual command sequences.
