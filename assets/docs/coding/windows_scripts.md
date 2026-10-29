# ADSMOD Windows Script Rules

Last updated: 2026-10-29

## Operational Script Expectations

- Use `start_on_windows.ps1` as the canonical launcher and maintenance entry point.
- Keep PowerShell usage explicit and deterministic.
- Preserve runtime-safe path handling.
- Route normal disposable runtime and tool caches below `runtimes/cache`; keep
  cleanup discovery limited to recognized application cache/build locations.
- Avoid destructive operations outside intended directories.
- Before cleanup or uninstall, inventory exact targets, reject unsafe roots,
  avoid following reparse points, and report skipped or failed paths.
- Preserve configured application data during standard uninstall; require a
  separate confirmed action for complete local-data removal.
- Keep script behavior compatible with the existing launcher and runtime flow.
- Load runtime endpoints through `server.runtime_config`, which reads `.env`
  and process environment; load database connection settings from
  `database.json` below the configured storage root. Application behavior
  settings remain in the selected database.
- Check dependency readiness before reinstalling runtimes or packages.

## Documentation Expectation

- When documenting operational scripts, provide CMD and PowerShell forms where they differ.
- Prefer existing repository scripts over ad-hoc manual command sequences.
