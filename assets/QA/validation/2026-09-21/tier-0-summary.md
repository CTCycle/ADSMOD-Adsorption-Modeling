# ADSMOD Tier 0 validation checkpoint

Last updated: 2026-09-21

Tier 0 is the campaign’s environment/startup gate. It is not complete yet.

| Slice | Status | Evidence |
| --- | --- | --- |
| `ADS-T0-01` CI executability | `PARTIAL` | [`ADS-T0-01/ci-workflow.md`](ADS-T0-01/ci-workflow.md) |
| `ADS-T0-02` Windows launcher lifecycle | `FAIL` | [`ADS-T0-02/launcher-lifecycle.md`](ADS-T0-02/launcher-lifecycle.md) |

The CI workflow’s zero-job failure is diagnosed and the minimal duplicate-key
fix is pushed. Run `35620065349` then proved all three jobs are created, while
exposing hosted `uv 0.11.30` Python provisioning and long Edge socket-path
failures; a follow-up CI-only remediation is in the working tree for the next
remote run. The official launcher’s normal start/readiness/browser/stop path
passed, but the controlled occupied-port scenario did not refuse startup as
required. Tier 1 remains unstarted until both slices are closed and the ledger
is updated against the repaired commit.
