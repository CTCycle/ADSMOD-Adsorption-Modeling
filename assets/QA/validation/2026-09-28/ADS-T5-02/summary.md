# ADS-T5-02 — residual repetition, restart, and cleanup recheck

Date: 2026-09-28
Tested revision: `17f247e` plus the scoped current-revision focus fix; the
final pushed SHA is recorded in the ledger and roadmap after commit.
Status: `PASS` for the bounded repetition/restart/cleanup scope below.

## Implementation and regression

- The import wizard now receives initial focus, contains Tab focus, closes on
  Escape, and restores focus to the Add dataset trigger.
- The file input has the visible accessibility name `Choose a dataset file`.
- The focused frontend regression covers initial focus, forward Tab, reverse
  Tab wrapping, Escape dismissal, and trigger focus restoration.
- `test_datasets_api.py` plus `test_fitting_api.py`: `16 passed`.
- Core jobs, ML jobs, and database restart persistence: `13 passed`.

## Serial soak evidence

The final serial run lasted `600.132 s` with a `15 s` cycle interval. It
performed `38` dataset preview/validate/commit/list/duplicate checks,
`37` fitting start/duplicate/poll-to-terminal cycles, and `37` delete/list
cleanup cycles. Dataset duplicate commits returned HTTP `400` `38/38` times;
dataset deletes returned `204` `37/37` times; the sentinel delete returned
`204`; all recorded fitting jobs reached `completed`; and `active_jobs` was
empty at the end. Persisted sentinel data contained `2` experiments and `21`
observations before its final `204` deletion.

Two controlled backend restarts used the same isolated SQLite storage:

- around local `20:41:32` (mid-run), process `22348` was stopped and process
  `39108` reached `/health/ready` and capabilities HTTP `200` before the next
  successful `201` commit;
- around local `20:46:11` (later-run), process `39108` was stopped and process
  `16792` reached `/health/ready` HTTP `200`; its first post-restart cycle
  recorded `201` commit, `400` duplicate, completed fitting, and `204` delete.

Three connection-reset/refused operations occurred in the restart gaps at
soak offsets `196.040 s`, `213.050 s`, and `485.138 s`. They are recorded as
controlled restart observations, not hidden product failures. Final cleanup
after the adjacent E2E run deleted all six remaining disposable datasets with
HTTP `204`, left `0` datasets and `0` active jobs, and retained only terminal
job history.

The structured final run summary is [`soak-results-two-restarts.json`](soak-results-two-restarts.json);
the restart/API/persistence reconciliation is [`restart-persistence.json`](restart-persistence.json)
and cleanup is [`job-cleanup.json`](job-cleanup.json). The reproducible runner
is [`soak_runner.py`](soak_runner.py).

## Runtime limitation

The official Windows launcher was exercised with isolated storage under
`assets/QA/validation/2026-09-28/ADS-T5-residual/session`; its UI/readiness
path passed, but its child repeatedly returned HTTP `500` on multipart commit.
The captured server cause was `sqlite3.OperationalError: attempt to write a
readonly database` from the host ACL on that QA directory. The same revision,
fixture, and API contract passed the clean managed `server.cli` lane and the
full serial soak using a writable isolated runtime. No source defect was
identified from the launcher-only ACL failure. This recheck remains bounded
and does not claim production-scale stress capacity.
