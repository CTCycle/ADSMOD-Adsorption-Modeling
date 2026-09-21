# ADS-T0-02 — official Windows launcher lifecycle

Date: 2026-09-21
Baseline: `develop` at `7ac893767b74` plus the uncommitted launcher/frontend
working-tree changes in this validation
Environment: Windows, PowerShell 7, Standard/Base profile, ports 6045/5173
Evidence strength: official launcher + Codex in-app browser + direct local HTTP
checks + process/port checks

## Scenarios

| Scenario | Result | Evidence |
| --- | --- | --- |
| Warm option-1 launch with current state | PASS | Official menu option 1 reported that dependency installation and Angular build were skipped. Backend and static preview readiness completed on 6045/5173. |
| Static SPA route | PASS | The Codex in-app browser loaded `/datasets`; the rendered page showed `Custom Datasets` and `Backend Online`. |
| Health and capabilities proxy | PASS | Direct local HTTP requests through `http://127.0.0.1:5173/health/ready` and `/api/v1/system/capabilities` returned successful JSON; `/datasets` returned HTTP 200. Chrome’s direct JSON navigation was separately blocked by `ERR_BLOCKED_BY_CLIENT`, so that JSON-only browser surface is not claimed as a browser-rendered pass. |
| Stop the launcher-owned application | PASS | Official stop/cleanup paths stopped the backend and preview processes; final listener checks reported 6045 and 5173 free. |
| Backend conflict, user declines | PASS | A controlled listener on 6045 was shown in the aggregate conflict prompt; answering `N` preserved it and no ADSMOD process was started. |
| Frontend conflict, user approves | PASS | A controlled listener on 5173 was approved once, terminated, both ports were rechecked, and the warm launch completed. |
| One PID owns both configured ports | PASS | One controlled PID appeared against both 6045 and 5173; the aggregate prompt listed one process and exactly one termination was issued. |
| Two distinct conflicting PIDs | PASS | Two controlled listeners produced one aggregate prompt listing two processes; each approved PID was terminated once and launch completed. |
| Early process failure reporting | WORKING | `Wait-ForHealth` now receives each started process, refreshes it during polling, and reports an early exit immediately; no induced crash was required for the warm lifecycle. |
| Dependency/build state repair and profile preservation | PARTIAL | Standard/Base repair, `npm ci`, build, state-file creation, and later warm reuse were observed. ML profile repair, package-lock invalidation, documentation-only invalidation, and deleted-build-state repair were not separately exercised in this campaign. |

## Automated checks

- `npm run lint`: PASS.
- `npm run test:unit`: PASS, 26 files / 68 tests.
- `npm run test:preview`: PASS, 3 tests covering static serving, SPA fallback,
  traversal rejection, proxying, and missing-build failure.
- `npm run build`: PASS.
- Launcher contract tests: PASS, 4 tests.
- PowerShell parser validation: PASS.
- `app\tests\run_tests.bat`: PARTIAL locally. Frontend checks completed, but
  the all-tests runner was started under the Base profile: four ML-dependent
  unit modules could not import `sklearn`/`keras`, and one Base-profile E2E
  navigation check could not find the ML-only `Training` link. The hosted
  three-job workflow was not rerun after this working-tree change.

## Remaining validation boundary

Owner-unresolved listeners, denied/failed termination, PID replacement during
the termination race, a new listener appearing during final recheck, package
lock invalidation, documentation-only invalidation, and ML-profile capability
regression remain unvalidated here. They are not classified as product
failures without a reproducible positive or negative result.

## Cleanup

Controlled listeners and launcher-owned processes created for this campaign
were stopped through the official launcher paths. The final check showed ports
6045 and 5173 free. The launcher was restored to the recorded Standard/Base
profile. Unrelated processes and repository cache contents were not terminated
or deleted.
