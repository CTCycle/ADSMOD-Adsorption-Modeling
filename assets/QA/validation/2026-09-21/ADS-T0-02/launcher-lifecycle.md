# ADS-T0-02 — official Windows launcher lifecycle

Date: 2026-09-21
Baseline: `develop` at `57baefed0b2be9e71cfa4f053226ac45ff2ffb6d`
Environment: Windows, PowerShell 7, base profile, ports 6045/5173
Evidence strength: official launcher + in-app browser + process/port checks

## Scenarios

| Scenario | Result | Evidence |
| --- | --- | --- |
| Start from the official `start_on_windows.ps1` menu | PASS | Menu option 1 reported that application environments/frontend build were ready and skipped dependency installation. Backend readiness completed at `http://127.0.0.1:6045/health/ready`; frontend readiness completed at `http://127.0.0.1:5173/`. |
| Browser page load and visible readiness | PASS | The Codex in-app browser loaded `/datasets`; the rendered page showed `Custom Datasets` and the footer showed `Backend Online`. A 1280×720 readiness screenshot was captured during the run. |
| Browser console errors/warnings | PASS | In-app browser console query returned no error or warning entries for the successful page. |
| Stop the launcher-owned application | PASS | Menu option 2 stopped the session-owned backend/frontend processes. A subsequent listener check reported no listener on 6045 or 5173. |
| Occupied backend port must be refused before startup | FAIL | A controlled local listener occupied 6045. The launcher did not emit the expected `Port 6045 is already in use` refusal; it attempted backend startup, waited the full readiness timeout, exited with code 1, and left the unowned listener intact. |

## Failure classification

The clean base launch, browser load, readiness gate, and owned-process stop are
working. The occupied-port protection scenario is a launcher defect or
launcher-environment interaction, not a remote-provider or ML block. The
observed code path is `Start-Application` → `Assert-PortAvailable` →
`Get-ListeningProcess`; the exact reason the preflight did not observe the
controlled listener is not yet isolated. Do not infer that the guard is safe
from its source presence alone.

## Current status

`FAIL` for the complete `ADS-T0-02` slice; `PARTIAL` for the clean lifecycle
sub-scope. Track the occupied-port failure as `ISSUE-006` in the project
ledger. Tier 1 must not begin until the guard is surgically fixed or the
environment interaction is explained and the scenario passes on the supported
Windows path.

## Cleanup

The controlled listener and all launcher-owned processes created for this run
were stopped. Final checks showed ports 6045 and 5173 free. Unrelated Python
processes and repository cache contents were not terminated or deleted.
