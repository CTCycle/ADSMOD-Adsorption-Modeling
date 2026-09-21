# ADSMOD Tier 0 validation checkpoint

Last updated: 2026-09-21

Tier 0 is the campaign’s environment/startup gate. It is not complete yet.

| Slice | Status | Evidence |
| --- | --- | --- |
| `ADS-T0-01` CI executability | `PASS` | [`ADS-T0-01/ci-workflow.md`](ADS-T0-01/ci-workflow.md) |
| `ADS-T0-02` Windows launcher lifecycle | `FAIL` | [`ADS-T0-02/launcher-lifecycle.md`](ADS-T0-02/launcher-lifecycle.md) |

The CI workflow’s zero-job failure is diagnosed and the minimal duplicate-key
fix is pushed. Final run `35622186997` passed all three jobs, including both
backend profiles, generated contracts, and all seven frontend visual projects.
The official launcher’s normal start/readiness/browser/stop path passed, but the
controlled occupied-port scenario did not refuse startup as required. Tier 1
remains unstarted until that remaining first-tier slice is closed and the
ledger is updated against the repaired commit.
