# ADSMOD End-to-End UI & System Validation

Date: 2026-09-02

Runtime under test: official Windows launcher, Angular dev server `127.0.0.1:5173`, core API `127.0.0.1:6045`, and the configured core-only topology without the optional ML service.

## 1. Executive Summary

The live ADSMOD core workflows were validated through the browser and API: navigation, dataset import/persistence, metadata editing, experiment loading, model fitting, NIST collection, public-materials loading, and the core-only Training states. The frontend and backend quality gates passed, and the browser console remained free of errors and warnings during the exercised flows.

Three product issues were found and fixed during validation:

1. The Windows launcher started both services but could not open the frontend URL because PowerShell treated the URL as a process executable. The launcher now opens the URL explicitly through Explorer.
2. Fitting jobs returned complete model metrics from the backend, but the Fitting page only showed the activity-log summary. The page now renders the completion status, best model, counts, and per-model metrics.
3. A NIST experiment containing an unsupported uptake unit could abort the whole category fetch. The mapper now skips only that experiment, logs the reason, and reports the skipped count to the UI while preserving canonical units.

The remaining boundaries are explicit: native browser file-chooser automation was unavailable, the optional ML-positive workflows were not executable in the core-only runtime, the live NIST ping reported the external service as unreachable, and the training fixture lacks `adsorbate_SMILE` values required to build a training dataset. These are unverified or environment/data limitations, not false-positive application successes.

## 2. Environment and Method

- Browser: Chrome through the Codex in-app browser surface.
- Responsive fallback: Playwright screenshots at 1440x900 and 600x900 because the in-app browser does not expose viewport control.
- Dataset fixture: `app/tests/fixtures/sample_adsorption.csv`.
- API validation used the running core service and a disposable validation subset; the committed sample dataset was retained for downstream UI fitting verification.
- The configured local SQLite database is user-owned under `%LOCALAPPDATA%/ADSMOD`. Normal sandboxed backend writes were blocked by Windows ACLs; elevated validation was used only where required for the local database and was stopped afterward.

## 3. Tested Areas and Results

### Application startup and navigation

- Official `.\start_on_windows.ps1 -Action Launch` completed successfully after the launcher fix and reported both backend and frontend services ready.
- `/` redirected to `/datasets`.
- `/datasets`, `/public-data`, `/public-materials`, `/dashboards`, `/fitting`, `/training`, `/training/processing`, `/training/datasets`, `/training/checkpoints`, and `/training/dashboard` loaded without browser console errors.
- Public-data and public-materials showed the core backend Online state.
- Dashboards rendered the current explicit placeholder rather than an empty or misleading data view.

### Dataset import and metadata

- Direct API preview of `sample_adsorption.csv` returned 200 with 21 rows, 6 columns, atomic confidence `0.95`, detected groups, and detected units.
- Validation with decimal separator `.` returned 200 and produced 2 experiments / 21 observations.
- Commit returned 201 and created the retained `sample_adsorption` dataset.
- The browser rendered the persisted dataset row and selected it successfully.
- Dataset metadata was edited in the UI: tags `qa, e2e` and description `Imported by the end-to-end validation run.` The PATCH returned 200 and the updated row was confirmed by a subsequent API list.
- Native file selection in the upload wizard was attempted but the in-app browser did not expose a working file-chooser event; see Section 6.

### Fitting

- Dataset `sample_adsorption`, Experiment A, and its 14 observations were loaded through the live UI/API path.
- Starting all 9 catalog models completed with 8 successful fits and `jovanovic` as the best model; the backend poll returned 200 with a completed job.
- The updated UI rendered the completed result panel, best-model summary, 8/9 fitted count, 14 observations, and the RMSE / R² / AICc table. The failed Dubinin-Radushkevich row remained visibly failed with em dashes rather than fabricated metrics.
- Reset cleared the result panel and restored the ready state.
- Starting with no dataset selected was blocked in the UI with `[ERROR] Select one dataset.`

### NIST public-data collection

- External Ping returned a successful core response with `server_ok: false`; the UI honestly displayed `Adsorption experiments server is unreachable.`
- Category fetch at fraction `0.001` completed with job `aa39f8f0`: 40 requested, 40 fetched, 26 already local, 14 newly stored, and 14 skipped because the source measurements had no supported canonical-unit conversion.
- The UI displayed the updated activity details, including requested, newly fetched, local, and skipped counts. The persisted fraction remained `0.001` after route changes.
- Public-materials loaded with 2 adsorbates and 2 adsorbent materials after the validation run.

### Training and dashboards

- The processing wizard opened, advanced to review, and accepted the Build Dataset action.
- The service returned and displayed the honest error `Training data missing adsorbate_SMILE values.` for the available fixture.
- Training datasets and checkpoints correctly showed empty states with unavailable Open / Resume actions.
- The training dashboard remained idle with empty metrics while no ML-ready training dataset existed.

### Backend/API checks

- `GET /health/live`: 200.
- `GET /health/ready`: 200.
- `GET /api/v1/capabilities`: 200; datasets, NIST, fitting, machine learning, training, and checkpoints were advertised by the current runtime.
- Dataset list, experiment list, observation list, training source list, and fitting model catalog returned 200.
- The fitting model catalog contained 9 models.
- Fitting job polling returned 200 and completed with the expected partial-success result.
- Dataset cleanup removed only disposable validation datasets; the retained sample dataset remained available for the UI fitting flow.

### Automated quality gates

- Python browser/API E2E suite: 30 passed.
- Post-fix NIST API E2E suite: 6 passed.
- Backend focused unit slice (`test_nist_repository.py`, `test_data_processing.py`): 25 passed.
- Frontend unit suite: 22 files / 44 tests passed.
- Frontend lint and Angular migration verification: passed.
- Frontend production build: passed.
- Targeted Ruff checks for the NIST mapper, service, contract, and unit test: passed.
- Browser console: no error or warning entries during the validated flows.

## 4. Findings and Fixes

### F-01 — Official launcher could not open the frontend URL

- Severity: Medium — fixed.
- The launcher started the backend and frontend, then failed while passing `http://127.0.0.1:5173` directly to `Start-Process`.
- `start_on_windows.ps1` now invokes `explorer.exe` with the URL argument.
- Verification: the official launcher completed with `[OK] ADSMOD started successfully.` and both service endpoints were reachable.

### F-02 — Fitting metrics were returned but not rendered

- Severity: Medium — fixed.
- The backend returned the full fitting payload, while the page displayed only the activity-log line.
- `models-page.component.ts` now renders the returned result summary and accessible metrics table, with a focused component test covering the best model and metric values.
- Verification: the live UI showed `Jovanovic`, `8/9` models fitted, `14` observations, and the per-model metrics after a real fitting job.

### F-03 — Unsupported NIST measurement units aborted category mapping

- Severity: Medium — fixed.
- Some provider records use a volume-percent uptake basis that cannot be converted to the app's canonical uptake contract without a mass-basis input. Previously, one such record could raise out of the category mapping loop.
- The mapper now skips the unsupported experiment with a warning; the service returns `skipped_count` / `skipped_experiment_count`; and the UI reports skipped records.
- Verification: the focused unit test passed, the NIST API suite passed, and live job `aa39f8f0` completed while reporting 14 skipped records instead of failing.

### F-04 — External NIST service was unreachable

- Severity: External dependency limitation — not a core product defect.
- The Ping endpoint completed normally with `server_ok: false`, and the UI surfaced the outage as an informational status. The category fetch still completed from the available provider response path.

### F-05 — Training fixture has no adsorbate SMILES

- Severity: Validation-data limitation — not classified as a runtime defect.
- The processing wizard reached the build operation, which correctly rejected the available data with `Training data missing adsorbate_SMILE values.`

### F-06 — UI and runtime upload-extension declarations differ

- Severity: Low — deferred.
- The runtime configuration advertises CSV/XLS/XLSX, while the upload UI help/input also describes TXT/JSON. This can create a misleading expectation that the latter formats are accepted by the backend.

## 5. UI and UX Observations

- The primary shell, navigation, status bar, activity log, and asynchronous state messaging remained coherent across the tested routes.
- The fitting result panel now makes backend work legible instead of requiring users to infer success from a log sentence.
- The result table preserves failed-model state with dashes and an explicit status, avoiding fabricated metrics.
- At 1440x900, the dataset page rendered without clipping and showed the persisted sample row.
- At 600x900, the dataset cards/actions and fitting controls reflowed cleanly without horizontal page overflow.
- At 600x900, the fitting model cards remained readable and stacked appropriately.
- NIST's wide data table remains contained in its own scroll region on narrow screens.
- The Dashboard page is an explicit product placeholder; no dashboard data view is currently implemented.

## 6. Unverified or Deferred Concerns

- Native upload wizard: the in-app browser could not expose a usable file chooser, so the visible file-selection step was not completed interactively. API preview/validation/commit and downstream UI persistence were verified separately.
- ML-positive workflows: processing of a valid training dataset, actual training, checkpoint creation/resume, and populated dashboard metrics were not run because the optional ML service was not active and the available fixture lacked required SMILES values.
- NIST enrichment/index paths were not exercised beyond the category fetch because they depend on external availability and can create additional local records.
- No scientific or clinical quality approval is implied by the fitting result; this validation covers runtime, integration, and persisted-result behavior.
- The local sandbox could not write the configured SQLite database due to Windows ACLs. Elevated validation succeeded, and all elevated service processes were stopped after testing.

## 7. Recommended Next Actions

1. Run the upload wizard in a browser surface with native file-chooser support against a disposable QA database, covering preview -> mapping -> validation -> commit interactively.
2. Start the optional ML service with a valid SMILES-bearing dataset and repeat processing, training, checkpoint, resume, and dashboard-positive flows.
3. Align the runtime upload-extension allow-list with the UI's advertised formats, or narrow the UI copy to the formats actually supported.
4. Decide whether to replace the Dashboard placeholder with the next product view or keep the current explicit roadmap state.

## Evidence Files

- `e2e-datasets-1440x900.png`
- `e2e-datasets-600x900.png`
- `e2e-fitting-600x900.png`
- `e2e-ui-system-validation-2026-09-02.md`

## Delivery Commits

- `0c6774f` — `fix: surface fitting results and launcher URL`
- `4f23349` — `fix: keep NIST imports canonical when units are unsupported`
