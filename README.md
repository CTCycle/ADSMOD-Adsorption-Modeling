# ADSMOD Adsorption Modeling
[![Release](https://img.shields.io/github/v/release/CTCycle/ADSMOD-Adsorption-Modeling?display_name=tag)](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/releases)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-22.12.0-5FA04E?logo=node.js&logoColor=white)](https://nodejs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CTCycle Portfolio](https://img.shields.io/badge/CTCycle-Portfolio-58a6ff?style=flat-square)](https://ctcycle.github.io/CTCycle/)
[![CI](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/workflows/ci.yml?query=branch%3Adevelop)

## 1. Project Overview

ADSMOD is a comprehensive web application designed for the collection, management, and modeling of adsorption data. This project represents the evolution and unification of two predecessor projects: **ADSORFIT** and **NISTADS Adsorption Modeling** (the former name of this repository).

### Service and frontend split

- Backend split:
- `app/server/core_service` (non-ML API workflows)
- `app/server/ml_service` (training/ML workflows)
- `app/server/shared` (shared persistence and repository layer)
- Frontend:
- `app/client` (Angular UI for source, fitting, and training)
- `/api/training/*` is routed to the optional ML service; all other `/api/*` traffic is routed to the core service.

By merging the capabilities of these systems into a single, cohesive platform, ADSMOD provides a robust workflow for researchers and material scientists. The application allows users to:
- **Collect** adsorption isotherms from the NIST Adsorption Database.
- **Enrich** material data with chemical properties fetched from PubChem.
- **Build** curated, standardized datasets suitable for machine learning.
- **Train and Evaluate** deep learning models to predict adsorption behaviors.

The system is organized as a modern web application with a responsive user interface and a backend focused on data processing and machine learning tasks.

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

ADSMOD provides an automated menu-driven launcher and maintenance script for Windows users.

1. Navigate to the `ADSMOD` directory.
2. Run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1`.

**What this script does:**
- Downloads portable Python, uv, and Node.js runtimes into `runtimes/` (first run only).
- Installs backend dependencies into `app/server/.venv`.
- Installs frontend dependencies and builds the unified frontend bundle.
- Starts the unified local web backend and frontend preview.
- Exposes setup, test, cleanup, database initialization, and uninstall actions.

**First Run vs. Subsequent Runs:**
- On the **first run**, setup may take time because runtimes and dependencies are downloaded.
- On **subsequent runs**, launch is faster because setup is reused.

### 3.2 Manual Setup (Advanced)

If you prefer manual setup or are running outside the launcher workflow:
1. Install Python and Node.js.
2. For core-only usage, run `uv sync --package adsmod-core-service --group dev` from `app/server`.
3. For ML/training support, run `uv sync --package adsmod-ml-service --group dev` from `app/server`, or `uv sync --all-packages --group dev` when explicitly validating both services.
4. Install frontend dependencies in `app/client`.
5. Launch backend and frontend processes.

### Backend startup commands (Stage 1)

```cmd
set ADSMOD_ENABLE_ML=true
app\server\.venv\Scripts\python.exe -m uvicorn app.server.app:app --host 127.0.0.1 --port 6045
app\server\.venv\Scripts\python.exe -m uvicorn core_service.app:app --host 127.0.0.1 --port 8000
app\server\.venv\Scripts\python.exe -m uvicorn ml_service.app:app --host 127.0.0.1 --port 8001
cd app\client && npm run dev
```

## 4. How to Use

### 4.1 Launching the Application

**Windows:**
Run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1` and select **Launch application**. The launcher starts the unified backend and frontend preview, waits for both to respond, and opens the local web UI.

### 4.2 Mode Switching

Local web mode uses the runtime file:

- `settings/.env`

Adjust host/port and runtime backend values in that file when needed.

### 4.3 Operational Workflow and UI Snapshots

The application workflow is exposed through one frontend:
- `source`
- `fitting`
- `training`
The snapshots below were captured from the current `develop` build (`v2.3.0` release preparation) and are intended to show representative product states without duplication.

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

*Checkpoints view: review saved checkpoints and resume previous experiments.*

<img src="assets/figures/training-checkpoints.png" alt="Training - Checkpoints View" width="1000" />

*Training Dashboard view: track run progress and monitor key training metrics.*

<img src="assets/figures/dashboard.png" alt="Training - Dashboard View" width="1000" />

## 5. Setup and Maintenance

Run `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1` to access setup and maintenance actions:

- **Remove logs**: clears `.log` files under `app/resources/logs`.
- **Install or update dependencies**: prepares shared runtimes, backend dependencies, and the unified frontend.
- **Uninstall application**: removes local runtime and build artifacts while preserving settings, resources, the database, and user data.
- **Initialize database**: creates or resets the project database schema.
- **Clear cache**: removes Python bytecode caches and the uv cache.

### 5.1 Frontend Development Commands

From `app/client`:

```bash
npm install
npm run dev
npm run build
```

Frontend API base path defaults to `/api`; the Angular development server routes `/api/training/*` to the ML service and all other `/api/*` calls to the core service through `app/client/proxy.conf.cjs`.

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

`.env` runtime keys used by the launcher, tests, and frontend startup:

| Variable | Description |
|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind host and port. |
| `UI_HOST`, `UI_PORT` | Frontend host and port for local web mode and tests. |
| `KERAS_BACKEND`, `MPLBACKEND` | ML/scientific runtime backend configuration. |
| `RELOAD` | Uvicorn reload toggle for local development. |
| `BACKEND_LOGS_VISIBLE` | Set to `true` to show backend logs in a dedicated terminal; defaults to `true` when absent. |
| `VITE_API_BASE_URL` | Optional frontend API base path written into runtime config; same-origin `/api` is used by default. |

Single canonical runtime file:
- `settings/.env`

## 9. Development Status

This project is still under active development. It will be updated regularly, but you may encounter bugs, issues, or incomplete features. Tagged releases (currently v2.3.0) are stable for local evaluation and testing.

## 8. License

This project is licensed under the **MIT License**. See `LICENSE` for full terms.

