# ADS-T3-03 — NIST enrichment and normalization boundaries

Date: 2026-09-24
Baseline: `396e2e092d810ef5182713a563f1de69eb0b5eca`
Status: `PARTIAL`

## Exercised

The current mapper and repository behavior were rechecked with the focused
`test_nist_repository.py` and `test_public_data.py` suites (14 passed). Those
tests cover normalized canonical units and the unsupported-unit skip boundary.
The live experiments job requested 40 records; the provider returned 14, all
14 were skipped because measurements could not be mapped to canonical units,
and the local count remained 255. The visible counter and backend log agree
that unsupported records were reported rather than silently accepted. The log
identifies `% Volume Adsorbed` as unsupported and `wt%` records without a
positive adsorbate molar mass. No positive persistent record increase is
claimed.

## Incomplete coverage and active issue

Guest/host enrichment was not run. The browser tool denied further interaction
after the experiments job because its usage limit had been reached. Thus this
slice does not establish the positive guest/host enrichment or broader
provider-coverage behavior. `ISSUE-002` remains open: those 14 records are
outside the canonical dataset until a safe conversion basis is supported.
The skip-and-report path worked in the exercised run; the source coverage gap
remains a limitation rather than an observed mapper defect.

## Next action

Validate guest and host fetch/enrichment in the rendered NIST workflow after
browser access is available, then recheck persisted counts and provenance.
Keep `PARTIAL` until those positive paths are exercised; keep `ISSUE-002` open
until unsupported measurement bases can be converted safely or excluded by a
documented coverage policy.
