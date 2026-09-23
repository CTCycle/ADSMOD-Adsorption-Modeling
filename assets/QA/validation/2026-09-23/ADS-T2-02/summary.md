# ADS-T2-02 — Binary Excel import lifecycle

**Date:** 2026-09-23
**Status:** PASS
**Implementation commit:** `c876a063d3fcef4de1ea074aaebba861da4b2c4c`

## Scope and result

Validated actual `.xls` and `.xlsx` workbooks through the canonical import
path: preview, mapping, validation, save, persisted inspection, experiment
switching, reload, and deletion. The Base runtime now declares pinned
`openpyxl==3.1.5` and `xlrd==2.0.2` dependencies; the lockfile resolves and
checks cleanly. No parser defect was observed after those readers were
available.

Both workbook fixtures contain the same six-row, nine-column, two-experiment
adsorption data used for the canonical import checks. Unit coverage confirmed
six preview rows, nine columns, two experiments, six observations, and
normalized values including 10,000 Pa and 0.12 mol/kg. Browser coverage saved
each format, inspected three persisted rows in the selected experiment after
reload, received HTTP `204` on deletion, and confirmed absence after another
reload. The browser run asserted no console, page, or dataset API errors.

## Evidence

- [`sample_adsorption.xls`](../../../../../app/tests/fixtures/sample_adsorption.xls)
  and [`sample_adsorption.xlsx`](../../../../../app/tests/fixtures/sample_adsorption.xlsx)
  are actual binary workbook fixtures.
- [`test_excel_workbooks_preview_and_validate_as_canonical_datasets`](../../../../../app/tests/unit/test_canonical_adsorption_import.py)
  covers parsing and normalization for both extensions.
- The browser lifecycle is parameterized for `.xls` and `.xlsx` in
  [`test_dataset_import_browser.py`](../../../../../app/tests/e2e/test_dataset_import_browser.py).
- The in-app Browser visually confirmed the imported cards and persisted
  observation inspector at 1280×720. Playwright supplied files to the real
  browser file input because the in-app tool could not drive its hidden native
  file picker.

## Verification

- Focused import suite: **20 passed** (unit, API E2E, and browser E2E).
- Ruff passed on the three changed Python test modules.
- `uv lock --check --project app/server` and `git diff --check` passed.
- Tests used the official launcher and an isolated disposable database. SQLite
  `PRAGMA quick_check` returned `ok` before the temporary data/cache root was
  removed; ports 5173 and 6045 were free after launcher shutdown.

## Remaining scope

No remaining gap was found for `.xls` or `.xlsx` in this import slice. Broader
fitting and remaining Tier 2 work are not covered by this result.
