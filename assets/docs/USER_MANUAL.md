# ADSMOD User Manual

Last updated: 2026-04-08

This manual explains how to use ADSMOD for dataset ingestion, fitting, and model training.

## 1. Getting Started

### 1.1 Start the application (Windows launcher)

1. Open the repository folder.
2. Run `ADSMOD\start_on_windows.bat`.
3. Wait for backend/frontend startup.
4. Open the UI URL shown by the launcher (typically `http://127.0.0.1:<UI_PORT>`).

### 1.2 Runtime and ports

- Runtime host/port values are configured in `ADSMOD/settings/.env`.
- Changes to `.env` apply to local web mode and packaged desktop mode.

## 2. Main Navigation

The application has three main tabs:

- `source`: dataset upload and NIST data collection.
- `fitting`: adsorption model fitting.
- `training`: dataset processing, training runs, and checkpoints.

## 3. User Journeys

### 3.1 Journey A: Upload and fit a local dataset

1. Go to `source`.
2. Select a local `.csv` or `.xlsx` file.
3. Upload and confirm dataset summary.
4. Move to `fitting`.
5. Select the uploaded dataset.
6. Enable desired fitting models.
7. Set optimizer and max iterations.
8. Start fitting and monitor status/log output.

### 3.2 Journey B: Build from NIST and run fitting

1. Go to `source`.
2. Trigger NIST collection/index/fetch/enrich actions as needed.
3. Confirm status/progress completion.
4. Go to `fitting`.
5. Select NIST-based dataset option.
6. Start fitting and review generated outputs.

### 3.3 Journey C: Build training dataset and start training

1. Go to `training`.
2. Create or select a processed dataset configuration.
3. Run dataset build and wait for completion.
4. Start a new training run with desired settings.
5. Monitor progress and metrics from the dashboard.

### 3.4 Journey D: Resume training from a checkpoint

1. Go to `training`.
2. Open the checkpoints view.
3. Select a compatible checkpoint.
4. Resume training.
5. Monitor status and verify metric continuity.

## 4. Primary Commands

### 4.1 Launcher and maintenance

- Start app: `ADSMOD\start_on_windows.bat`
- Maintenance menu: `ADSMOD\setup_and_maintenance.bat`

### 4.2 Tests

- Full test runner: `tests\run_tests.bat`
- Direct pytest: `.\runtimes\.venv\Scripts\python.exe -m pytest tests -v`

### 4.3 Frontend workflow (from `ADSMOD/client`)

- Install deps: `npm install`
- Run dev server: `npm run dev`
- Build: `npm run build`

### 4.4 Desktop packaging

- Prepare runtimes: `ADSMOD\start_on_windows.bat`
- Build Tauri artifacts: `release\tauri\build_with_tauri.bat`

## 5. Usage Patterns

### 5.1 Recommended operating pattern

1. Set host/port/runtime values in `ADSMOD/settings/.env`.
2. Use launcher-managed runtimes (`runtimes/`) instead of global tooling.
3. Work in short runs: ingest -> validate -> fit/train -> review outputs.
4. Use checkpoints for long training sessions to avoid losing progress.

### 5.2 Job monitoring pattern

- Start long-running actions from UI controls.
- Poll status through UI progress areas.
- Cancel jobs when a run is misconfigured or no longer needed.
- Relaunch with corrected settings.

## 6. Key Features

- Local dataset upload (`.csv`, `.xlsx`).
- NIST-A data collection and enrichment.
- Multi-model adsorption fitting workflows.
- Training dataset composition and processing.
- Training start/resume with checkpoint support.
- Dashboard-style progress and status tracking.
- Windows launcher and desktop packaging support.

## 7. Troubleshooting

- Application does not start:
  - rerun `ADSMOD\start_on_windows.bat` and review console output.
- UI cannot reach backend:
  - verify `FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, `UI_PORT` in `.env`.
- Tests fail due to missing optional dependencies:
  - set `OPTIONAL_DEPENDENCIES=true`, run launcher again, then rerun tests.
- Desktop packaging fails:
  - ensure Rust `cargo` and an active toolchain are installed.
