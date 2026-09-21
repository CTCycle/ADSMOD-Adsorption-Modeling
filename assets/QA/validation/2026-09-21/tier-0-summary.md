# ADSMOD Tier 0 validation checkpoint

Last updated: 2026-09-21

Tier 0 is the campaign’s environment/startup gate. It is complete for the
validated baseline below; Tier 1 is the next untested campaign tier.

| Slice | Status | Evidence |
| --- | --- | --- |
| `ADS-T0-01` CI executability | `PASS` | [`ADS-T0-01/ci-workflow.md`](ADS-T0-01/ci-workflow.md) |
| `ADS-T0-02` Windows launcher lifecycle | `PASS` | [`ADS-T0-02/launcher-lifecycle.md`](ADS-T0-02/launcher-lifecycle.md) |

The CI workflow’s zero-job failure is diagnosed and the minimal duplicate-key
fix is pushed. Final run `35622186997` passed all three jobs, including both
backend profiles, generated contracts, and all seven frontend visual projects.
The official launcher’s clean start/readiness/browser/stop/relaunch path passed.
Controlled listeners on ports 6045 and 5173 were refused before application
startup after the active-listener fallback was added, and final listener checks
were free. Tier 0 is closed; Tier 1 remains the next untested campaign tier.
