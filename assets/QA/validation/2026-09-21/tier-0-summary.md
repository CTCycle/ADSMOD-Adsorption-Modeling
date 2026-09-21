# ADSMOD Tier 0 validation checkpoint

Last updated: 2026-09-21

Tier 0 is the campaign’s environment/startup gate. It is not complete yet.

| Slice | Status | Evidence |
| --- | --- | --- |
| `ADS-T0-01` CI executability | `PARTIAL` | [`ADS-T0-01/ci-workflow.md`](ADS-T0-01/ci-workflow.md) |
| `ADS-T0-02` Windows launcher lifecycle | `FAIL` | [`ADS-T0-02/launcher-lifecycle.md`](ADS-T0-02/launcher-lifecycle.md) |

The CI workflow’s zero-job failure is diagnosed and the minimal duplicate-key
fix is pushed. Run `35621291855` attempt 2 proved all three jobs are created,
the frontend ordinary checks pass, and all seven visual projects pass; both
backend jobs still fail because the hosted interpreter was not installed for
`3.14.7`. The next CI-only remediation adds `actions/setup-python@v5` before
uv. The official launcher’s normal start/readiness/browser/stop path passed,
but the controlled occupied-port scenario did not refuse startup as required.
Tier 1 remains unstarted until both slices are closed and the ledger is updated
against the repaired commit.
