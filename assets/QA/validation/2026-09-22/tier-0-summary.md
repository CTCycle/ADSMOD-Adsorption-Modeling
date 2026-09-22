# ADSMOD Tier 0 closure summary

Date: 2026-09-22
Validated implementation SHA: `2eb3bc13823bedae1be1791fc980a78613d00ef1`
Hosted CI: [run 35725502460](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35725502460)
Status: `Tier 0 CLOSED`

## Result

| Slice | Final status | Evidence |
| --- | --- | --- |
| `ADS-T0-01` | `PASS` | Exact hosted `base-backend`, `ml-backend`, and `frontend` jobs passed; `test:preview` and all 7 Edge projects ran successfully. |
| `ADS-T0-02` | `PASS` | Official lifecycle, real listener approval/decline, full deterministic conflict matrix, stale-state matrix, ML-profile preservation, early process-failure cleanup, and final port/process checks passed. |

## Fixes applied

- Kept port ownership fail-closed when process identity cannot be resolved and
  rechecked PID, process name, and UTC start time immediately before
  termination, closing the PID-reuse race.
- Made `app/tests/run_tests.bat` profile-correct: it detects Base/ML, requires
  Development tooling for the comprehensive runner, and excludes only
  ML-positive Base tests rather than installing optional ML packages.
- Added regression guards for the launcher identity contract and profile-aware
  runner behavior.

## Final environment and cleanup

The official launcher was restored to `Standard / Base`. The final fault-free
launch → browser `/datasets` → proxy readiness/capabilities → stop → warm
relaunch → stop sequence passed. Ports 6045 and 5173 were free afterward, no
launcher-owned backend or preview process remained, and unrelated processes,
persistent data, and pre-existing cache residue were preserved.

Tier 1 remains unstarted and is the next validation boundary.
