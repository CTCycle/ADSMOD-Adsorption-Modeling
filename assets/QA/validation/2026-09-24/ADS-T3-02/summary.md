# ADS-T3-02 — NIST status, indexing, and fetch lifecycle

Date: 2026-09-24
Baseline: `396e2e092d810ef5182713a563f1de69eb0b5eca`
Status: `PARTIAL`

## Exercised

On the isolated database copy, the rendered Sources view reached NIST status
and showed 433 cached NIST records. The experiments, guest, and host provider
ping actions all reported that their servers were reachable. Index actions
completed and reported 39,988 available experiments, 455 guests, and 9,328
hosts.

The experiments fetch used a fraction of `0.001`. The UI reported 40 requested,
14 fetched, 255 local records, and 14 skipped for lack of canonical units. The
backend log confirms `requested=40`, `fetched=14`, `local=255`, followed by
`skipped 14 of 14 fetched records`; the local count remained 255, so no
positive persisted growth is established. The log records unsupported `%
Volume Adsorbed` and `wt%` measurements that lacked a positive adsorbate molar
mass. See [`../ADS-T3-01/backend.stderr.log`](../ADS-T3-01/backend.stderr.log)
and [`../ADS-T3-03/summary.md`](../ADS-T3-03/summary.md).

## Incomplete coverage

Guest and host fetch jobs were not run. Further browser interaction was denied
by Codex browser auto-review after the experiments fetch, with the tool
reporting that the usage limit had been reached. This is an access/tooling
boundary, not a failed NIST response; no workaround was attempted. The
experiments fetch completed before that denial. As a result, the full NIST job
lifecycle is not validated and this slice remains `PARTIAL`.

## Supporting checks

The focused backend public-data/NIST suite passed 14 tests, the NIST frontend
unit suite passed 3 tests, focused Ruff passed, frontend lint passed, and the
production frontend build completed. See
[`../ADS-T3-01/summary.md`](../ADS-T3-01/summary.md) for commands, build output,
isolation, source-database integrity, and cleanup evidence.

## Next action

Complete guest and host fetch through the rendered workflow on an isolated
database when in-app browser access is available. Confirm resulting local
counts and persisted records before upgrading this slice to `PASS`.
