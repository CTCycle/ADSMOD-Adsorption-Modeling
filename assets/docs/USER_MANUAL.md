# ADSMOD User Manual

Last updated: 2026-04-24

## 1. Start the Application

### Windows launcher (recommended)

CMD:

```cmd
ADSMOD\start_on_windows.bat
```

PowerShell:

```powershell
.\ADSMOD\start_on_windows.bat
```

The launcher prepares runtimes/dependencies, starts backend and frontend, and opens the UI URL from `ADSMOD/settings/.env`.

## 2. Main Navigation

The app has three top-level sections:

- `source`: load local datasets and run NIST collection.
- `fitting`: configure optimization and run adsorption model fitting.
- `training`: process datasets, manage checkpoints, and run/resume training.

## 3. Core Workflows

### A. Upload and fit a local dataset

1. Open `source`.
2. Upload `.csv`, `.xls`, or `.xlsx`.
3. Confirm dataset stats.
4. Open `fitting`.
5. Select dataset, model set, optimizer, and iterations.
6. Start fitting and monitor logs.

### B. Use NIST data for fitting

1. Open `source`.
2. Run NIST category actions (ping/index/fetch/enrich) as needed.
3. Confirm status updates.
4. Open `fitting`.
5. Select `NIST-A Collection`.
6. Start fitting and monitor job status.

### C. Build training data and run training

1. Open `training`.
2. In `Data Processing`, build processed dataset(s).
3. In `Train datasets`, start a new training run.
4. Use `Training Dashboard` to monitor progress, metrics, and logs.

### D. Resume from checkpoint

1. Open `training` -> `Checkpoints`.
2. Select checkpoint.
3. Resume training with additional epochs.
4. Validate resumed metrics on dashboard.

## 4. Common Commands

### Launch and maintenance

- Start app: `ADSMOD\start_on_windows.bat`
- Maintenance menu: `ADSMOD\setup_and_maintenance.bat`

### Tests

- Full runner: `tests\run_tests.bat`
- Direct pytest: `.\runtimes\.venv\Scripts\python.exe -m pytest tests -v`

### Frontend dev (from `ADSMOD/client`)

- `npm install`
- `npm run dev`
- `npm run build`

### Desktop packaging

- `release\tauri\build_with_tauri.bat`

## 5. Troubleshooting

- Backend/UI unreachable:
  - Check `FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, `UI_PORT` in `ADSMOD/settings/.env`.
- Missing test dependencies:
  - Set `OPTIONAL_DEPENDENCIES=true` in `.env`, rerun launcher, then rerun tests.
- Packaging failure:
  - Ensure Rust/Cargo toolchain is installed and active.
