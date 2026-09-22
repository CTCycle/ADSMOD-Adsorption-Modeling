# ADS-T0-02 — official Windows launcher lifecycle and failure matrix

Date: 2026-09-22
Implementation SHA: `2eb3bc13823bedae1be1791fc980a78613d00ef1`
Environment: Windows, PowerShell 7, Python 3.14.7, Node 22.13.0, ports 6045/5173
Final profile: official `Standard / Base`; ML proof used official `Development / ML` and was restored
Evidence strength: official launcher + Codex in-app browser + direct local HTTP + isolated process/port harnesses

## Normal lifecycle

| Scenario | Result | Evidence |
| --- | --- | --- |
| Official Standard/Base install | PASS | Menu option 4 recorded `Standard / Base`, synchronized the locked runtime, installed frontend dependencies, and built the bundle. |
| First warm option-1 launch | PASS | Launcher reported that dependency installation and Angular build were skipped; backend and static preview became ready. |
| Browser `/datasets` | PASS | The rendered in-app browser page showed `Custom Datasets` and `Backend Online`. |
| Frontend proxy health/capabilities | PASS | `/health/ready` through port 5173 returned `state=ready`; `/api/v1/system/capabilities` returned HTTP 200 with `features.machine_learning=false`. |
| Official stop | PASS | Launcher stopped its backend and preview PIDs. |
| Warm relaunch and second stop | PASS | The same skip path and readiness completed again; official stop cleaned both owned processes. |
| Final cleanup | PASS | Ports 6045 and 5173 were free; no ADSMOD backend/preview process remained; the state file still recorded `Standard / Base`. |

## Port and process-safety matrix

The launcher’s conflict functions were exercised in an isolated harness using
controlled process objects and current listener/identity snapshots. These tests
are branch guards, not substitutes for the live listener checks below.

| Scenario | Result | Required behavior observed |
| --- | --- | --- |
| Clean ports | PASS | Launch continues without conflict work. |
| Owner cannot be resolved | PASS | Launch aborts before dependency/build/startup work and kills nothing. |
| Non-interactive conflict | PASS | Launch aborts without terminating anything. |
| User declines | PASS | Launch cancels and kills nothing. |
| Process disappears before termination | PASS | Race is treated safely and no replacement is targeted. |
| Listener unresolved immediately before kill | PASS | Launch aborts and kills nothing. |
| PID/process identity changes after approval | PASS | Name/start-time mismatch is detected and launch aborts. |
| Termination fails or is denied | PASS | Failure is reported and startup does not continue. |
| Process remains after bounded wait | PASS | Existing 5-second bounded wait fails closed. |
| New process acquires the port after termination | PASS | Final port recheck detects the listener and aborts. |
| One approved PID owns both ports | PASS | Exactly one termination attempt is issued. |
| Multiple approved PIDs | PASS | Each approved PID is terminated at most once. |

Live official-menu smoke checks added real Windows evidence:

- One controlled `pwsh` PID listened on both 6045 and 5173. The launcher
  displayed one aggregate owner, approval issued one termination, both ports
  were rechecked, and the normal application launch completed.
- A separate controlled listener on 6045 was declined. The listener remained
  present and no ADSMOD process was started.

The surgical launcher fix keeps the existing PID, process-name, and process
start-time identity contract and adds a final `Refresh()`/identity comparison
immediately before `Kill($true)`. Listener ownership is now unresolved unless
both process name and start time are readable.

## Dependency and build-state invalidation

The matrix ran in a disposable local clone with stubbed install/build/start
actions and the real fingerprint/state predicates. The clone was removed after
validation.

| Scenario | Result | Observed action |
| --- | --- | --- |
| Everything current | PASS | No backend install, `npm ci`, or build. |
| `package-lock.json` changed | PASS | Locked frontend dependency repair and build. |
| Frontend source changed | PASS | Build only. |
| Documentation/excluded config changed | PASS | No dependency repair or build. |
| Build state deleted | PASS | One build and state recreation. |
| Built index missing | PASS | Build before preview startup. |
| Frontend dependency state missing/corrupt | PASS | Locked frontend dependency repair. |
| Backend fingerprint stale | PASS | Recorded profile repair, including `Development / ML`. |
| State schema/version invalid | PASS | Stale state repair instead of trust. |
| Second launch after repair | PASS | Warm-reuse path; no repeated repair/build. |

## ML-profile preservation

The official launcher installed `Development / ML` and recorded that profile in
`.venv/.adsmod-dependency-state.json`. After a deliberate backend fingerprint
stale marker, option 1 reported repair of the recorded `Development / ML`
profile and synchronized ML dependencies. The running application then showed:

- `/api/v1/system/capabilities` through the frontend proxy with
  `features.machine_learning=true`;
- ML training capability `true`;
- HTTP 200 for `/api/v1/training/configuration`; and
- a rendered browser Training route with the training navigation visible.

A second ML launch skipped synchronization. The official launcher then restored
the developer’s original `Standard / Base` profile, and the Base capability
regression returned `machine_learning=false` with ML packages absent.

The initial ML download attempt hit Windows socket error 10013 while reaching
the package index. This was classified as an external network environment
failure; the same official flow succeeded with network access and did not
require a product change.

## Early process failures

An isolated readiness harness used real short-lived PowerShell child processes
and the current `Wait-ForHealth` implementation:

- Backend child exited with code 17 before readiness. Failure was reported and
  cleanup completed in about 2.5 seconds, without the 60-second timeout.
- Frontend child exited with code 23 before readiness. Failure was reported and
  both the already-started backend and frontend were cleaned up in about 2.3
  seconds.

Both ports were free after each case. No artificial sleep workaround was added.

## Profile-correct test runner

The Base mismatch was reproduced: ML-positive route assertions and ML-only unit
imports cannot run in a Base environment. `app/tests/run_tests.bat` now reads
the dependency-state feature and installation profiles, rejects a Standard
environment before starting services because the comprehensive runner requires
Development tooling, and excludes only the ML-positive Base unit/E2E modules.

- Development/Base: `106 passed, 1 deselected` with the live services skipped.
- Development/ML: the exact non-E2E CI suite passed `150 tests`.
- Standard/Base: an explicit early error named the missing Development
  tooling; it did not install ML packages or start the application.

## Adjacent regressions and cleanup

`npm run test:preview` passed 3/3; frontend lint, unit, production build, and
all 7 Edge layout projects passed. PowerShell parsing and launcher static
contract tests passed. The final official profile is `Standard / Base`, the
frontend built entry is `dist/browser/index.html`, both ports are free, and no
launcher-owned process remains. Pre-existing ACL-protected cache directories
and unrelated user data were preserved.

Chrome’s direct navigation to JSON was separately blocked by
`ERR_BLOCKED_BY_CLIENT`; the browser claim is limited to the visible rendered
`/datasets` page, while proxy JSON was verified by direct local HTTP.
