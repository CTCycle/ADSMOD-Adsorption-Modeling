# ADS-T3-02 — NIST status, indexing, and fetch lifecycle

Date: 2026-09-24

Code under test: working-tree changes based on `223dc5862cb453f5171155af5486490965a6cdd9`
Status: `PASS`

## Exercised

Started the official Windows launcher with a fresh isolated `%LOCALAPPDATA%`
profile and exercised the rendered Public Data → Sources workflow in the
Codex in-app browser at `http://127.0.0.1:5173/public-data/sources` (1280×720).
The configured application database was not used or modified.

The three NIST pings reported reachable. Index actions reported 39,988
experiments, 455 guests, and 9,328 hosts. With fraction `0.001`, the rendered
workflow fetched 1 guest and 10 hosts. Repeating both category fetches returned
zero new records, leaving their local source-record totals at 1 and 10.

The experiments fetch requested and received 40 records. The UI reported 26
local records and 14 skipped records without canonical units. SQLite confirms
26 persisted isotherms and 26 NIST adsorption provenance records. The 14 skips
are tracked under `ISSUE-002` and `ADS-T3-03`.

After a browser reload, the acquisition view still showed 26/39,988
experiments, 8/455 guests, and 14/9,328 hosts (48 local category records in
total). The larger guest/host totals include entities linked to the fetched
experiments as well as the directly fetched category records. The provider card
reported 37 cached NIST provenance records: 26 adsorption, 1 chemical, and 10
material records.

## Persistence and integrity

The isolated database passed `PRAGMA quick_check` and was at Alembic revision
`20260924_fitting_cancel`. Its NIST source-record composition and experiment
count match the rendered totals. The original configured database SHA-256
remained `5446E0BAB31479D86D7A4C80EA5D2349008C06B13F0CC70E97FA7284F65E7722`.

The live fetch exposed a defect in category counts and duplicate filtering:
standalone NIST guest/host source records were not included by repository
queries. The repository now includes direct NIST source links alongside
experiment-linked records. The regression tests cover these counts, identifier
sets, and standalone reference-material loading. See
[`../ADS-T3-03/summary.md`](../ADS-T3-03/summary.md) for the associated
enrichment repair.

## Supporting checks

`test_nist_repository.py` and `test_public_data.py`: 15 passed. Ruff passed for
the changed repository, service, and repository tests. `git diff --check`
passed. Logs are in [`nist-public-data-tests.log`](nist-public-data-tests.log)
and [`nist-ruff.log`](nist-ruff.log); the rendered state is captured in
[`browser-state.md`](browser-state.md).

## Remaining scope

This pass validates bounded fractions and reachable provider responses; it
does not establish completeness or stable availability of the full external
NIST catalog. Unsupported experiment measurements remain explicitly tracked by
`ISSUE-002`. Positive PubChem public-data provider behavior and COD importing
remain separate `UNTESTED` gates (`ADS-T3-04` and `ADS-T3-05`).
