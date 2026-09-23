# ADS-T2-03 — Dataset import input boundaries

**Date:** 2026-09-23
**Status:** PASS
**Implementation commit:** `c876a063d3fcef4de1ea074aaebba861da4b2c4c`

## Scope and result

Rechecked current parser/API behavior for empty files, corrupt Excel uploads,
missing required mappings, malformed measurement values, unsupported file
extensions, and mismatched parser input arrays. Invalid content stayed
client-visible and did not create a dataset. No product defect was observed in
the exercised boundaries.

The API E2E checks confirmed:

- Empty upload returns HTTP `400` with `Uploaded dataset is empty.`
- Corrupt `.xls` and `.xlsx` inputs each return HTTP `400`.
- A mapping without required measurement/material roles returns a validation
  result marked `invalid`, with the missing-role issue codes.
- A malformed pressure value returns an `invalid_row` validation issue.
- Existing parser-unit coverage continues to reject unsupported extensions and
  mismatched input arrays.

## Verification

- `test_datasets_api.py` passed with its existing positive import/list/delete
  coverage and the new negative-boundary cases.
- `test_canonical_adsorption_import.py` passed its parser-boundary regressions
  and Excel preview/validation checks.
- Combined focused import suite: **20 passed**.
- Ruff passed on the three changed Python test modules.
- `uv lock --check --project app/server` and `git diff --check` passed.
- API tests ran against the official launcher's isolated data root. The
  disposable SQLite database passed `PRAGMA quick_check` and was removed after
  the launcher stopped and ports 5173/6045 were verified free.

## Remaining scope

No remaining gap was found in the covered import boundaries. The broader Tier
2 fitting slices `ADS-T2-04`–`ADS-T2-06` remain untested; the public NIST,
positive ML training, and dashboard-scope limitations remain separate gates.
