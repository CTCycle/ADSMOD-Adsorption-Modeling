# ADS-T5-02 — repetition, restart, and cleanup validation

Date: 2026-09-26
Validation base: `bcc81e824f62fbbc13a8059a8b989b9ce31f8d25` plus the scoped
validation tests in this campaign
Status: `PASS` for the bounded scope below

## Evidence

- The full dataset API E2E file passed `11` tests against the isolated live
  backend, including the new repeated-import regression.
- The first import created dataset `repeat_adsorption_6c23775c` as canonical ID
  `2`, with `2` experiments and `21` observations. Repeating the identical
  commit returned HTTP `400` with the explicit duplicate message and the list
  contained exactly one matching dataset; no second canonical record was
  created.
- The rendered Custom Datasets page showed the imported dataset before and
  after backend restart. The persisted values after restart remained ID `2`,
  `2` experiments, and `21` observations.
- A live fitting run across nine model keys started as job `cc6dd30e`. An
  immediate second submission for the same fitting lane returned HTTP `400`
  with `A fitting job is already running.` The first job reached `completed`.
- The completed fitting run remained available as persisted run `1` after a
  second backend restart. The restarted in-memory job list was empty, proving
  no active job bookkeeping survived the restart as stale work.
- The focused backend suite passed `42` tests, including success/failure/
  cancellation cleanup and the new repeated-terminal-job assertion that
  `threads`, stop events, processes, and job configs are empty after three
  terminal jobs.
- Final cleanup stopped the isolated backend and preview processes and showed
  no listeners on ports `6045` or `5173`.

The structured restart and job evidence is in
[`restart-persistence.json`](restart-persistence.json) and
[`job-cleanup.json`](job-cleanup.json). Rendered observations are in
[`browser-state.md`](browser-state.md).

## Remaining limitation

This is a bounded resilience pass, not a load or long-duration stress test.
The existing fitting cancellation path and process-job lifespan coverage were
also retained as adjacent regression coverage; no new failure-injection stress
campaign was added.
