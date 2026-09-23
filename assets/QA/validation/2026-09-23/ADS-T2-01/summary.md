# ADS-T2-01 — CSV dataset import lifecycle

**Date:** 2026-09-23
**Status:** PASS
**Implementation commit:** `c876a063d3fcef4de1ea074aaebba861da4b2c4c`

## Scope and result

Validated the current CSV workflow through preview, mapping, validation, save,
persisted inspection, experiment switching, reload, and deletion on a
disposable database. No product defect was observed in this slice.

The browser flow used the existing six-row multi-experiment fixture
[`adsmod-dataset-csv-20260916.csv`](../../../../../assets/QA/adsmod-dataset-csv-20260916.csv).
The app detected nine columns, validated two experiments and six observations,
and displayed three persisted observations after switching to the second
experiment. Reload restored the saved dataset and its inspection state. The
DELETE request returned `204`, and a fresh page load confirmed the deleted
dataset remained absent.

## Rendered and browser evidence

The in-app Browser rendered the dataset workspace at 1280×720. The imported
dataset card showed two experiments and six observations; the inspection table
showed the selected experiment's three persisted rows with source rows 5–7,
pressures 15,000, 30,000, and 60,000 Pa, and uptakes 0.1, 0.19, and 0.31 mol/kg.
The automated Chromium flow asserted no browser-console errors, page errors, or
dataset API errors.

The in-app Browser could not operate the hidden native file picker. The
repository browser test supplied the same file through the real HTML file input
with Playwright `set_input_files`, then exercised the rendered workflow.

## Verification

- `test_csv_import_persists_inspection_experiment_switch_and_deletion` passed
  as part of the focused browser suite.
- Focused import suite: **20 passed** (unit, API E2E, and browser E2E).
- Ruff passed on the three changed Python test modules.
- `uv lock --check --project app/server` and `git diff --check` passed.
- The official Windows launcher was used with an isolated `LOCALAPPDATA` data
  root. After shutdown, SQLite `PRAGMA quick_check` returned `ok`; ports 5173
  and 6045 were free. The isolated database and caches were removed.

## Remaining scope

No remaining gap was found within `ADS-T2-01`. Excel format coverage is recorded
separately in [`ADS-T2-02`](../ADS-T2-02/summary.md); invalid-input coverage is
recorded in [`ADS-T2-03`](../ADS-T2-03/summary.md).
