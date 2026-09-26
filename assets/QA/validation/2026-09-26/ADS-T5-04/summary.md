# ADS-T5-04 bounded performance validation

Last updated: 2026-09-26

Status: PASS for the bounded fixture scope.

## Scope and evidence

The isolated measurement harness in measure_t5_04.py used the canonical
sample_adsorption.csv fixture and a TemporaryDirectory storage root. It
validated backend readiness, CSV preview/validation/commit, experiment lookup,
five local Public Data listing endpoints, one fitting job polled to a terminal
completed state, and dataset deletion cleanup. The run produced two
experiments, 21 observations, three public sources, and a completed fitting
job after four status requests. The server advertised a one-second polling
interval; the harness completed in 355.621 ms because it used a bounded
50-ms measurement loop for the isolated job.

The focused backend suite passed 40/40 tests, including the real .xls/.xlsx
parser boundaries, provider retry/error contracts, persistence-safe Public
Data re-import behavior, and the bounded SQL round-trip guard. The SQL guard
was also selected directly and passed 1/1. The frontend polling interval
helper suite passed 8/8.

See metrics.json for timings and counts. All measured state was disposable
and no application process or listener was left running.

## Limitations

This is a bounded correctness/performance slice, not a long-duration load or
stress campaign. Live provider latency and upstream availability remain
covered by ADS-T5-01 and are not inferred from the local database timings.
