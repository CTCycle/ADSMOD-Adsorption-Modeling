# ADSMOD Architecture

Last updated: 2026-04-24

## 1. System Overview

ADSMOD is a Windows-first local application with:

- Backend: FastAPI (`ADSMOD/server`), served by Uvicorn.
- Frontend: React 18 + TypeScript + Vite (`ADSMOD/client`).
- Optional desktop shell: Tauri (`ADSMOD/client/src-tauri` and `release/tauri`).
- Runtime/toolchain bootstrap: portable Python, uv, Node.js under `runtimes/`.

## 2. Directory and File Structure

Generated folders (`__pycache__`, `node_modules`, `dist`, Tauri `target`) are excluded below.

```text
.
|- ADSMOD/
|  |- client/
|  |  |- src/
|  |  |  |- components/
|  |  |  |  |- ControlsPanel.tsx
|  |  |  |  |- DatasetBuilderCard.tsx
|  |  |  |  |- DatasetProcessingWizard.tsx
|  |  |  |  |- EquationRenderer.tsx
|  |  |  |  |- InfoModal.css
|  |  |  |  |- InfoModal.tsx
|  |  |  |  |- MarkdownRenderer.tsx
|  |  |  |  |- ModelCard.tsx
|  |  |  |  |- ModelConfigForm.tsx
|  |  |  |  |- NewTrainingWizard.tsx
|  |  |  |  |- NistCollectionRows.tsx
|  |  |  |  |- ResumeTrainingWizard.tsx
|  |  |  |  |- Sidebar.tsx
|  |  |  |  |- TrainingSetupRow.tsx
|  |  |  |  |- UIComponents.tsx
|  |  |  |  |- WizardNavigationFooter.tsx
|  |  |  |  |- WizardProgressIndicator.tsx
|  |  |  |- features/training/
|  |  |  |- hooks/
|  |  |  |- pages/
|  |  |  |  |- ConfigPage.tsx
|  |  |  |  |- MachineLearningPage.tsx
|  |  |  |  |- ModelsPage.tsx
|  |  |  |- services/
|  |  |  |  |- datasetBuilder.ts
|  |  |  |  |- datasets.ts
|  |  |  |  |- fitting.ts
|  |  |  |  |- http.ts
|  |  |  |  |- index.ts
|  |  |  |  |- jobs.ts
|  |  |  |  |- nist.ts
|  |  |  |  |- training.ts
|  |  |  |- adsorptionModels.ts
|  |  |  |- App.tsx
|  |  |  |- constants.ts
|  |  |  |- index.css
|  |  |  |- main.tsx
|  |  |  |- types.ts
|  |  |  |- vite-env.d.ts
|  |  |  |- wizard-styles.css
|  |  |- src-tauri/
|  |  |  |- src/main.rs
|  |  |  |- tauri.conf.json
|  |  |- package.json
|  |  |- vite.config.ts
|  |- resources/
|  |- scripts/
|  |- server/
|  |  |- api/
|  |  |  |- datasets.py
|  |  |  |- entrypoint.py
|  |  |  |- fitting.py
|  |  |  |- nist.py
|  |  |  |- training.py
|  |  |- common/
|  |  |  |- constants.py
|  |  |  |- utils/
|  |  |- configurations/
|  |  |  |- environment.py
|  |  |  |- management.py
|  |  |  |- startup.py
|  |  |- domain/
|  |  |- learning/
|  |  |  |- inference/
|  |  |  |- models/
|  |  |  |- training/
|  |  |- repositories/
|  |  |  |- database/
|  |  |  |- queries/
|  |  |  |- schemas/
|  |  |  |- serialization/
|  |  |- services/
|  |  |  |- data/
|  |  |  |- modeling/
|  |  |  |- fitting.py
|  |  |  |- job_responses.py
|  |  |  |- jobs.py
|  |  |  |- training.py
|  |  |- app.py
|  |- settings/
|  |  |- .env
|  |  |- configurations.json
|  |- setup_and_maintenance.bat
|  |- start_on_windows.bat
|- assets/docs/
|- release/tauri/
|  |- build_with_tauri.bat
|  |- scripts/
|  |  |- clean-tauri-build.ps1
|  |  |- export-windows-artifacts.ps1
|- runtimes/
|  |- .venv/
|  |- nodejs/
|  |- python/
|  |- uv/
|  |- uv.lock
|- tests/
|  |- backend/
|  |- e2e/
|  |- fixtures/
|  |- server/
|  |- unit/
|  |- conftest.py
|  |- run_tests.bat
|- pyproject.toml
```

## 3. Application Entry Points

- Backend ASGI app: `ADSMOD/server/app.py` (`app = FastAPI(...)`).
- Backend runtime command: `python -m uvicorn ADSMOD.server.app:app`.
- Frontend entry: `ADSMOD/client/src/main.tsx`.
- Frontend shell: `ADSMOD/client/src/App.tsx`.
- Windows launcher: `ADSMOD/start_on_windows.bat`.
- Desktop runtime entry: `ADSMOD/client/src-tauri/src/main.rs`.
- Test orchestrator: `tests/run_tests.bat`.

## 4. Backend Layering and Module Responsibilities

### Layered flow

`endpoint (api/*) -> service (services/*) -> repository/serializer (repositories/*) -> persistence/runtime`

### Key modules

- `server/api/*`: HTTP contract and exception translation.
- `server/services/fitting.py`: fitting job orchestration.
- `server/services/training.py`: dataset build and training lifecycle orchestration.
- `server/services/data/nist_service.py`: async NIST and PubChem workflows.
- `server/services/jobs.py`: cross-cutting job manager (thread/process execution, status, cancel).
- `server/repositories/database/*`: SQLite/PostgreSQL backends and initialization.
- `server/repositories/schemas/models.py`: SQLAlchemy schema and constraints.
- `server/learning/*`: training runtime internals (model, worker, scheduler, metrics).

## 5. API Endpoints

All functional routes are mounted under `/api` in `server/app.py`.

| Method | Endpoint | Responsibility |
|---|---|---|
| GET | `/api/health` | Health probe |
| POST | `/api/datasets/load` | Upload and load dataset file |
| GET | `/api/datasets/names` | List dataset names |
| GET | `/api/datasets/by-name/{dataset_name}` | Fetch dataset payload by name |
| POST | `/api/fitting/run` | Start fitting background job |
| GET | `/api/fitting/nist-dataset` | Build/load NIST dataset for fitting |
| GET | `/api/fitting/jobs` | List fitting jobs |
| GET | `/api/fitting/jobs/{job_id}` | Get fitting job status |
| DELETE | `/api/fitting/jobs/{job_id}` | Cancel fitting job |
| POST | `/api/nist/fetch` | Start combined NIST fetch job |
| POST | `/api/nist/properties` | Start NIST properties enrichment job |
| GET | `/api/nist/status` | Aggregated NIST data status |
| GET | `/api/nist/categories/status` | Per-category NIST status |
| POST | `/api/nist/categories/{category}/ping` | Ping category upstream endpoint |
| POST | `/api/nist/categories/{category}/index` | Fetch category index |
| POST | `/api/nist/categories/{category}/fetch` | Fetch category records by fraction |
| POST | `/api/nist/categories/{category}/enrich` | Enrich guest/host properties |
| GET | `/api/nist/jobs` | List NIST jobs |
| GET | `/api/nist/jobs/{job_id}` | Get NIST job status |
| DELETE | `/api/nist/jobs/{job_id}` | Cancel NIST job |
| GET | `/api/training/datasets` | Training dataset availability summary |
| GET | `/api/training/dataset-sources` | List training dataset sources |
| DELETE | `/api/training/dataset-source` | Delete dataset source |
| POST | `/api/training/build-dataset` | Start dataset build job |
| GET | `/api/training/processed-datasets` | List processed datasets |
| GET | `/api/training/dataset-info` | Read processed dataset metadata |
| DELETE | `/api/training/dataset` | Clear one/all processed datasets |
| GET | `/api/training/jobs` | List training dataset-build jobs |
| GET | `/api/training/jobs/{job_id}` | Get dataset-build job status |
| DELETE | `/api/training/jobs/{job_id}` | Cancel dataset-build job |
| GET | `/api/training/checkpoints` | List checkpoints |
| GET | `/api/training/checkpoints/{checkpoint_name}` | Checkpoint details |
| DELETE | `/api/training/checkpoints/{checkpoint_name}` | Delete checkpoint |
| POST | `/api/training/start` | Start training session |
| POST | `/api/training/resume` | Resume from checkpoint |
| POST | `/api/training/stop` | Request training stop |
| GET | `/api/training/status` | Poll training state and metrics |

## 6. Data Persistence

- Primary default: embedded SQLite at `ADSMOD/resources/database.db`.
- Optional external DB: PostgreSQL when `embedded_database=false` in `ADSMOD/settings/configurations.json`.
- ORM: SQLAlchemy schema in `server/repositories/schemas/models.py`.
- Checkpoints: filesystem under `ADSMOD/resources/checkpoints`.
- Runtime config:
  - process/env: `ADSMOD/settings/.env`
  - application/database/job defaults: `ADSMOD/settings/configurations.json`

## 7. Async vs Sync Behavior

- FastAPI includes both sync and async handlers.
- NIST workflows are async (`httpx` calls) in `NISTDataService`, then executed in background jobs (thread mode) through sync wrappers in `api/nist.py`.
- Fitting jobs run as synchronous compute tasks inside the job manager thread runner.
- Training runs in a separate process (`ProcessWorker`) and reports progress back to in-memory state and job status.
- Cancellation model is cooperative first (`stop_event` / `should_stop`), with forced process termination after timeout for process jobs.

## 8. Frontend Architecture

- Main shell (`App.tsx`) hosts three top-level pages:
  - `source` -> `ConfigPage`
  - `fitting` -> `ModelsPage`
  - `training` -> `MachineLearningPage`
- API integration is centralized in `client/src/services/*`.
- API base path is normalized to local path-only values (`VITE_API_BASE_URL`, default `/api`) in `client/src/constants.ts`.
- UI state for long-running operations is poll-driven (`jobs.ts`, `training.ts`, `nist.ts`, `fitting.ts`).
