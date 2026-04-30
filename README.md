# ADSMOD Adsorption Modeling
[![Release](https://img.shields.io/github/v/release/CTCycle/ADSMOD-Adsorption-Modeling?display_name=tag)](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/releases)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-22.12.0-5FA04E?logo=node.js&logoColor=white)](https://nodejs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/workflows/ci.yml)

## 1. Project Overview

ADSMOD is a comprehensive web application designed for the collection, management, and modeling of adsorption data. This project represents the evolution and unification of two predecessor projects: **ADSORFIT** and **NISTADS Adsorption Modeling** (the former name of this repository).

By merging the capabilities of these systems into a single, cohesive platform, ADSMOD provides a robust workflow for researchers and material scientists. The application allows users to:
- **Collect** adsorption isotherms from the NIST Adsorption Database.
- **Enrich** material data with chemical properties fetched from PubChem.
- **Build** curated, standardized datasets suitable for machine learning.
- **Train and Evaluate** deep learning models to predict adsorption behaviors.

The system is organized as a modern web application with a responsive user interface and a backend focused on data processing and machine learning tasks.

> **Work in Progress**: This project is still under active development. It will be updated regularly, but you may encounter bugs, issues, or incomplete features.

## 2. Model and Dataset

This project utilizes deep learning techniques to model adsorption phenomena.

- **Model**: The core learning capability is based on the **SCADS** model architecture.
- **Learning**: The system relies on **Supervised Learning**, using historical experimental data to train predictive models.
- **Dataset**:
  - **Primary Source**: Experimental adsorption isotherms from the **NIST Adsorption Database**.
  - **Enrichment**: Chemical properties (for example molecular weights and SMILES strings) from **PubChem**.
  - The application handles fetch, cleanup, and merge steps to produce training-ready datasets.

## 3. Installation

### 3.1 Windows (One Click Setup)

ADSMOD provides an automated installation and launcher script for Windows users.

1. Navigate to the `ADSMOD` directory.
2. Run `start_on_windows.bat`.

**What this script does:**
- Downloads portable Python, uv, and Node.js runtimes into `runtimes/` (first run only).
- Installs backend dependencies from `pyproject.toml` into `app/server/.venv`.
- Installs frontend dependencies and builds the frontend bundle.
- Starts backend and frontend.

**First Run vs. Subsequent Runs:**
- On the **first run**, setup may take time because runtimes and dependencies are downloaded.
- On **subsequent runs**, launch is faster because setup is reused.

### 3.2 Manual Setup (Advanced)

If you prefer manual setup or are running outside the launcher workflow:
1. Install Python and Node.js.
2. Install backend dependencies from the app/server.
3. Install frontend dependencies in `app/client`.
4. Launch backend and frontend processes.

## 4. How to Use

### 4.1 Launching the Application

**Windows:**
Double-click `start_on_windows.bat`. This launches backend/frontend and opens the UI at `http://<UI_HOST>:<UI_PORT>` from `settings/.env`.

**Windows (Packaged Tauri App):**
Build with `release\tauri\build_with_tauri.bat`, then launch from `release/windows/installers` or `release/windows/portable`.

### 4.2 Mode Switching

Both local web mode and packaged Tauri mode use the same runtime file:

- `settings/.env`

Adjust host/port and runtime backend values in that file when needed.

### 4.3 Operational Workflow and UI Snapshots

The application workflow is organized into three top navigation tabs: `source`, `fitting`, and `training`.
The snapshots below were captured from the current `develop` build (`v2.2.0` release preparation) and are intended to show representative product states without duplication.

#### 4.3.1 Data Source Configuration

- Upload local `.csv` or `.xlsx` adsorption data.
- Collect and enrich adsorption data from NIST-A.
- Monitor ingestion and enrichment progress from the UI.

<img src="assets/figures/home.png" alt="Source Page - Data Source Configuration" width="1000" />

*Source tab: upload local datasets, review sample/size metadata, and run NIST-A collection tools.*

#### 4.3.2 Models and Fitting

- Select a dataset (uploaded or NIST).
- Configure optimizer settings and fitting iterations.
- Select adsorption models and run fitting.
- Review fit status and logs.

<img src="assets/figures/fitting.png" alt="Fitting Page - Models and Optimization" width="1000" />

*Fitting tab: choose adsorption models, configure optimization, and inspect fitting logs.*

#### 4.3.3 Training

- Build machine-learning-ready datasets.
- Configure and start new training experiments.
- Resume previous runs from checkpoints.
- Monitor run status and metrics from the dashboard.

*Train Datasets view: pick a processed dataset and launch a training setup.*

<img src="assets/figures/training-datasets.png" alt="Training - Train Datasets View" width="1000" />

*Data Processing view: compose and manage machine-learning-ready datasets.*

<img src="assets/figures/training-data-processing.png" alt="Training - Data Processing View" width="1000" />

*Checkpoints view: review saved checkpoints and resume previous experiments.*

<img src="assets/figures/training-checkpoints.png" alt="Training - Checkpoints View" width="1000" />

*Training Dashboard view: track run progress and monitor key training metrics.*

<img src="assets/figures/dashboard.png" alt="Training - Dashboard View" width="1000" />

## 5. Setup and Maintenance

Run `setup_and_maintenance.bat` to access setup and maintenance actions:

- **Remove logs**: clears `.log` files under `app/resources/logs`.
- **Uninstall app**: removes local runtimes and build artifacts while preserving folder scaffolding.
- **Initialize database**: creates or resets the project database schema.
- **Clean desktop build artifacts**: removes Tauri build output under release targets.

### 5.1 Frontend Development Commands

From `app/client`:

```bash
npm install
npm run dev
npm run build
```

Frontend API base path is controlled by `VITE_API_BASE_URL` (default: `/api`).

## 6. Resources

The application stores data and artifacts in specific directories:

- **checkpoints**: trained model weights, training history, and model configuration files.
- **database**: local SQLite database for metadata, cached responses, and experiment indexes.
- **logs**: application logs for troubleshooting and monitoring.
- **runtimes**: portable Python/uv/Node.js downloaded by the Windows launcher.
- **runtime venv**: backend virtual environment at `app/server/.venv`.
- **runtime lockfile**: backend lockfile at `app/server/uv.lock`.
- **templates**: starter assets such as the `.env` scaffold.

## 7. Configuration

Runtime/process values are loaded from `settings/.env`.

Database mode and backend defaults are loaded from `settings/configurations.json`.

`.env` runtime keys used by launcher/tests/frontend/Tauri startup:

| Variable | Description |
|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind host and port. |
| `UI_HOST`, `UI_PORT` | Frontend host and port for local web mode and tests. |
| `KERAS_BACKEND`, `MPLBACKEND` | ML/scientific runtime backend configuration. |
| `RELOAD` | Uvicorn reload toggle for local development. |
| `OPTIONAL_DEPENDENCIES` | Enables optional test dependencies in launcher flow. |
| `VITE_API_BASE_URL` | Frontend API base path (same-origin `/api` by default). |

Single canonical runtime file:
- `settings/.env`

## 8. License

This project is licensed under the **MIT License**. See `LICENSE` for full terms.

## Final validation and green-run requirement

After completing the restructure and all rewiring, run the full validation process in one uninterrupted pass until everything is green.

Do not stop to ask for confirmation between steps. If a check fails, diagnose the issue, fix it, and rerun the relevant checks until the project passes. Only stop if there is a hard blocker that cannot be resolved from the repository contents, and in that case report the exact blocker, the failing command, the error output, and the files likely involved.

Validation must include, where applicable:

* Run and verify all root-level `.bat` launcher scripts.
* Run and verify all root-level `.bat` maintenance scripts.
* Confirm every `.bat` script still works from the repository root after the restructure.
* Confirm runtime discovery and setup works with `runtimes/`.
* Run `uv sync` from `app/server/`.
* Confirm `.venv` and `uv.lock` are created or used only inside `app/server/`.
* Run backend linting, formatting checks, type checks, and tests if present.
* Run the FastAPI backend startup path and confirm imports, settings loading, resources loading, and path constants work.
* Generate or validate `app/shared/openapi.json` from the backend.
* Run frontend install, linting, type checks, tests, and production build if present.
* Confirm frontend API generation uses `app/shared/openapi.json` where applicable.
* Run all test suites under `app/tests/`.
* Validate Python imports after the move.
* Validate frontend imports after the move.
* Validate all script references under `app/scripts/`.
* Validate documentation and README references to the new layout.
* Validate Docker builds or compose files if present.
* Validate Tauri configuration if present.
* Run Tauri development checks if present.
* Run Tauri packaging or release build if present, and confirm output still lands in `release/` or the configured release output location.
* Confirm `.github/` remains unchanged.
* Confirm `assets/` and `release/` contents remain unchanged.
* Confirm `settings/` and `app/resources/` were extracted correctly and are read from their new locations.
* Confirm there are no stale references to the old root-level backend, client, tests, settings, or resources paths.

At the end, provide a concise validation report listing:

* Every command run.
* Whether it passed or failed.
* Any fixes made after failures.
* Any remaining warnings or limitations.
* Final green status once all required checks pass.
