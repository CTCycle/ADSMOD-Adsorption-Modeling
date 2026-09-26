# ADS-T5-05 documentation and contract reconciliation

Last updated: 2026-09-26

Status: PARTIAL.

## Passed checks

- The generated configuration schema exactly matched the tracked canonical
  schema by SHA-256.
- The tracked OpenAPI snapshot passed its complete-surface test.
- The focused contract/configuration suite passed 14/14 tests.
- The new fail-closed generator guard passed 5/5 focused contract tests.
- Ruff passed for the changed generator and contract test.
- A targeted documentation scan covered 31 Markdown documents and 248
  relative links with zero missing targets.
- The stale-reference scan found no obsolete active API, runtime, or cache
  references. The remaining matches are intentional current source-state
  wording and historical architecture provenance.

## Remaining limitation

The current Windows environment is the Base backend profile and reports that
Torch and RDKit are unavailable. Generating OpenAPI from that profile produced
41 paths, while the canonical ML-inclusive snapshot contains 56 paths,
including 15 training/checkpoint paths. The generator now exits with code 1
before writing output under Base, and the canonical snapshot remained
unchanged. A current-revision positive regeneration of the 56-path snapshot
still requires the ML-enabled development profile; the supported hosted CI
lane is the appropriate recheck.

The operator and quality-gate documentation now states this profile
requirement explicitly. This slice therefore remains PARTIAL rather than
claiming a current-host full OpenAPI regeneration pass.
