# ADS-T3-03 — NIST enrichment and normalization boundaries

Date: 2026-09-24

Code under test: working-tree changes based on `223dc5862cb453f5171155af5486490965a6cdd9`
Status: `PARTIAL`

## Exercised

The live experiments fetch on the isolated current-code profile requested and
received 40 records. The rendered activity reported 26 local records and 14
skipped for unsupported canonical units. The database contains 26 NIST
isotherms. The focused `test_public_data.py` suite and repository regressions
passed, including the mapper's unsupported-unit boundary. The previous live
backend capture identifies the affected bases as `% Volume Adsorbed` and
`wt%` records without a positive adsorbate molar mass;
[`../ADS-T3-01/backend.stderr.log`](../ADS-T3-01/backend.stderr.log) preserves
that diagnostic evidence.

Guest enrichment ran for the one fetched guest: 1 name requested, 1 matched,
and 1 row updated. Host enrichment ran for the 10 fetched hosts: 10 names
requested, 0 matched, and 0 rows updated. This is a valid no-match outcome for
that bounded sample, not evidence of a host provider failure. The sample does
not establish a positive host enrichment.

Validation exposed a second NIST data-path gap: standalone category records
were saved with NIST provenance but omitted from the repository's local counts,
duplicate checks, and enrichment inputs. The fix now queries both NIST
experiment-linked entities and standalone NIST source-linked entities. Tests
verify that those entities are counted once, their identifiers participate in
duplicate filtering, and their records are available to enrichment. See
[`nist-public-data-tests.log`](../ADS-T3-02/nist-public-data-tests.log).

The final rendered state survived browser reload: 26/39,988 experiments,
8/455 guests, and 14/9,328 hosts. The database is healthy, contains 37 NIST
source records (26 adsorption, 1 chemical, 10 material), and leaves the
configured database unchanged. The exact UI capture is in
[`../ADS-T3-02/browser-state.md`](../ADS-T3-02/browser-state.md).

## Why this slice remains partial

`ISSUE-002` remains open. The 14 unsupported measurements are still excluded
because the available unit/metadata basis does not permit a safe conversion.
No conversion rule or broader coverage policy was inferred. The host sample
also produced no positive PubChem match, so host property enrichment is not
demonstrated on this provider sample. These limits are distinct from the
observed guest positive result and from the confirmed 26 persisted experiment
records.

The NIST guest path calls the legacy `PubChemClient`; this run does not validate
the separate Public Data PubChem resolution/structure workflow. `ADS-T3-04`
therefore remains `UNTESTED`, as does positive COD import in `ADS-T3-05`.

## Supporting checks

`test_nist_repository.py` and `test_public_data.py`: 15 passed. Ruff passed for
the changed repository, service, and repository tests. `git diff --check`
passed. See [`../ADS-T3-02/nist-public-data-tests.log`](../ADS-T3-02/nist-public-data-tests.log)
and [`../ADS-T3-02/nist-ruff.log`](../ADS-T3-02/nist-ruff.log).

## Next action

Resolve the safe conversion or explicit coverage-policy decision for `ISSUE-002`
and recheck representative unsupported measurements. Keep this slice
`PARTIAL` until the remaining unit-coverage gap is handled. Then select the
separate positive PubChem and COD gates with their own persistence and
provenance evidence.
