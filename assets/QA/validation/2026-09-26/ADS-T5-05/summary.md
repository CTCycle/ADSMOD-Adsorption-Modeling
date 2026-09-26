# ADS-T5-05 documentation and contract reconciliation

Last updated: 2026-09-26

Status: PASS.

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
- Hosted CI run `36266365072` passed all three jobs on implementation SHA
  `15294177748ffa30929a15394df4361ff94254d7`. Its ML-enabled backend job
  generated the canonical OpenAPI and configuration-schema snapshots and
  passed the exact `git diff` check.

## Local profile boundary

The current Windows environment is the Base backend profile and reports that
Torch and RDKit are unavailable. Generating OpenAPI from that profile produced
41 paths, while the canonical ML-inclusive snapshot contains 56 paths,
including 15 training/checkpoint paths. The generator now exits with code 1
before writing output under Base, and the canonical snapshot remained
unchanged. This is an expected local profile boundary; the supported hosted
ML-enabled lane completed the positive current-implementation regeneration.

The operator and quality-gate documentation now states this profile
requirement explicitly. T5-05 is therefore PASS with the Base-profile
limitation documented above.
