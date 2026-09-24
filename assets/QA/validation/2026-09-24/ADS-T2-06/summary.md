# ADS-T2-06 — Fitting cancellation and recovery

- **Date:** 2026-09-24
- **Result:** PASS
- **Environment:** Windows, official `start_on_windows.ps1` launcher, isolated temporary data root, 200,000-observation persisted fixture
- **Code baseline:** working tree based on `9fde82b` (`develop`)

## Scope and findings

An initial live cancellation exposed a schema defect: the SQLite check
constraint `ck_fitting_runs_status` rejected the terminal status `cancelled`,
so the API returned HTTP 500. Added Alembic revision ID
`20260924_fitting_cancel` in
[`20260924_fitting_cancellation.py`](../../../../../app/server/migrations/versions/20260924_fitting_cancellation.py)
to permit the status and updated the canonical SQLAlchemy constraint. Official
launcher startup applied the migration and reported it as the database head.

The rerun exercised the cancellation button against a long-running fit using
the 200,000-observation fixture. The fixture generator is retained as
[`prepare_cancel_dataset.py`](prepare_cancel_dataset.py). The UI cancellation request succeeded; the
persisted run reached `cancelled`, returned `Fitting cancelled.`, and retained
zero results. A second start while a job was active was rejected with HTTP 400
and `A fitting job is already running.`. A subsequent fit started and reached
`completed`, showing that the job lock and cancellation state were released.

The job manager now passes cooperative cancellation events into fitting,
prevents duplicate active fitting jobs, and records cancellation atomically.
The fitting service and repository coordinate the running run transition.

## Automated checks

- Focused backend suite — 20 passed, including job cancellation, fitting
  metrics, database initialization, and restart persistence; see
  [`backend-focused.log`](backend-focused.log).
- `app/tests/e2e/test_fitting_api.py` — 5 passed with the official server
  active; see [`fitting-api-e2e.log`](fitting-api-e2e.log).
- Frontend unit suite — 26 files, 72 tests passed, including cancel control and
  persisted-result restoration coverage; see
  [`ADS-T2-05/frontend-unit.log`](../ADS-T2-05/frontend-unit.log).
- Frontend lint/build, Ruff, Python formatting, and `git diff --check` passed.

Both pytest logs contain a non-blocking warning that the repository's existing
`runtimes/cache/pytest` cache path could not be written. A combined repeat
attempted the API E2E checks after the official server had been stopped and
therefore got `ECONNREFUSED`; the diagnostic is retained in
[`api-e2e-post-shutdown-precondition.log`](api-e2e-post-shutdown-precondition.log).
The corrected run with the official server active passed all five API E2E
checks.

## Limitations

No remaining T2-06 gate. Cancellation is cooperative; it is checked at the
fitting/optimizer callback boundary. The exercised 200,000-observation case
proved cancellation and subsequent recovery in this local runtime.
