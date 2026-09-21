# ADS-T0-02 — official Windows launcher lifecycle

Date: 2026-09-21
Baseline: `develop` at `57baefed0b2be9e71cfa4f053226ac45ff2ffb6d`
Remediation: active TCP-listener fallback in `start_on_windows.ps1` at `f68d082c5bd724296d45cc0726e9d79c6719603e`
Environment: Windows, PowerShell 7, base profile, ports 6045/5173
Evidence strength: official launcher + in-app browser + process/port checks

## Scenarios

| Scenario | Result | Evidence |
| --- | --- | --- |
| Start from the official `start_on_windows.ps1` menu | PASS | Menu option 1 reported that application environments/frontend build were ready and skipped dependency installation. Backend readiness completed at `http://127.0.0.1:6045/health/ready`; frontend readiness completed at `http://127.0.0.1:5173/`. |
| Browser page load and visible readiness | PASS | The Codex in-app browser loaded `/datasets`; the rendered page showed `Custom Datasets` and the footer showed `Backend Online`. A 1280×720 readiness screenshot was captured during the run. |
| Browser console errors/warnings | PASS | In-app browser console query returned no error or warning entries for the successful page. |
| Stop the launcher-owned application | PASS | Menu option 2 stopped the session-owned backend/frontend processes. A subsequent listener check reported no listener on 6045 or 5173. |
| Occupied backend port must be refused before startup | PASS | A controlled local listener occupied 6045. After the active-listener fallback, the official menu refused immediately with `Port 6045 is already in use, but its owning process could not be resolved`; the unowned listener was preserved. |
| Occupied frontend port must be refused before startup | PASS | A controlled local listener occupied 5173. The official menu refused before starting either application process with the corresponding safe refusal; the unowned listener was preserved. |

## Failure classification

The clean base launch, browser load, readiness gate, owned-process stop, clean
relaunch, and both occupied-port protections pass. The original failure was a
launcher-environment interaction: `Get-NetTCPConnection` returned no owner in
the interactive preflight even though an active TCP listener was present. The
smallest fix adds an
`[System.Net.NetworkInformation.IPGlobalProperties]::GetActiveTcpListeners()`
cross-check and fails closed when the owner PID cannot be resolved. The
launcher still never terminates the unowned listener.

## Current status

`PASS` for the complete `ADS-T0-02` slice on the supported Windows path. The
active-listener fallback was exercised through the official menu for both
6045 and 5173, and the adjacent clean start/browser/stop/relaunch regression
also passed. `ISSUE-006` is resolved and moved to the ledger’s historical
findings; Tier 1 is now the next untested campaign tier.

## Cleanup

Both controlled listeners and all launcher-owned processes created for these
runs were stopped. Final checks showed ports 6045 and 5173 free. Unrelated
processes and repository cache contents were not terminated or deleted.
