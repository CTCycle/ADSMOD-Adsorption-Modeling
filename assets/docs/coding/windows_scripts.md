# ADSMOD Windows Script Rules

Last updated: 2026-08-02

## Operational Script Expectations

- Use `start_on_windows.ps1` as the canonical launcher and maintenance entry point.
- Keep PowerShell usage explicit and deterministic.
- Preserve runtime-safe path handling.
- Avoid destructive operations outside intended directories.
- Keep script behavior compatible with the existing launcher and runtime flow.
- Keep the launcher’s canonical paths aligned with `app/resources/adsmod.json`;
  dependency readiness is checked before reinstalling runtimes or packages.

## Documentation Expectation

- When documenting operational scripts, provide CMD and PowerShell forms where they differ.
- Prefer existing repository scripts over ad-hoc manual command sequences.
