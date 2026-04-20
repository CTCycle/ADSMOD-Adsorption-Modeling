# ADSMOD Architecture

Last updated: 2026-04-20

ADSMOD is a local adsorption-modeling application composed of:
- a Python FastAPI backend (`ADSMOD/server`),
- a React + TypeScript frontend (`ADSMOD/client`),
- launcher-managed local runtimes (`runtimes`).

## 1. Repository Structure

- `ADSMOD/`: application code.
  - `server/`: backend API, domain models, services, repositories, training runtime.
    - `api/`: route modules (`datasets`, `entrypoint`, `fitting`, `nist`, `training`).
    - `services/`: orchestration (`jobs.py`, `job_responses.py`, `fitting.py`, `training.py`), plus data/modeling services.
      - `services/modeling/nist_dataset.py`: fitting NIST dataset preparation and normalization.
    - `repositories/`: persistence and query helpers.
      - `serialization/normalization.py`: shared serialization normalization/conversion helpers.
      - `serialization/data.py`: repository serializer and persistence wiring.
    - `services/data/`: NIST clients/builders (`nistads.py`) and service orchestration (`nist_service.py`).
    - `learning/`: model training runtime components.
  - `client/`: frontend app.
    - `src/pages/`: top-level pages (`ConfigPage`, `ModelsPage`, `MachineLearningPage`).
    - `src/components/`: shared UI and workflow widgets.
    - `src/features/training/`: training-related feature modules.
    - `src/services/`: API and polling helpers.
  - `settings/`: runtime configuration (`.env`, `configurations.json`).
  - `resources/`: runtime data (database, checkpoints, logs, templates).
  - `start_on_windows.bat`: runtime bootstrap + launch flow.
  - `setup_and_maintenance.bat`: maintenance actions.
- `tests/`: `e2e`, `unit`, `server`, and `backend/performance` suites.
- `runtimes/`: local Python, uv, Node.js, and `.venv`.
- `release/`: Tauri build/export scripts and Windows artifacts.

## 2. Runtime Topology

- Backend process: Uvicorn serving `ADSMOD.server.app:app`.
- Frontend process:
  - launcher mode: built frontend served locally,
  - test mode: static server from `client/dist`.
- API exposure:
  - health and optional direct routers,
  - mirrored `/api/...` routes for same-origin frontend calls.

## 3. Backend Layering

1. API layer (`server/api`): request/response schemas and route handlers.
2. Service layer (`server/services`): orchestration and workflows.
3. Repository layer (`server/repositories`): data access.
4. Learning/runtime layer (`server/learning`): training and checkpoint runtime behavior.

`server/api/fitting.py` and `server/api/training.py` are intentionally thin route modules:
- they validate request/response contracts,
- delegate business logic to `server/services`,
- translate service exceptions into HTTP errors.

## 4. Key Subsystems

### 4.1 Data ingestion and preparation
- Uploaded datasets are loaded via dataset API routes.
- NIST data fetch/index/enrich flows are exposed by NIST routes.
- Training dataset composition/build is managed through training services and jobs.

### 4.2 Fitting and training
- Fitting runs as background jobs with status polling.
- Fitting route orchestration is owned by `server/services/fitting.py`.
- Fitting NIST dataset preparation is executed in `server/services/modeling/nist_dataset.py`.
- Training supports fresh runs and resume-from-checkpoint flows.
- Checkpoint compatibility is validated against runtime metadata.
- Training runtime internals remain under `server/learning`, while route-facing orchestration is owned by `server/services/training.py`.

### 4.3 Job orchestration
- Centralized in `ADSMOD/server/services/jobs.py` (`job_manager`).
- Supports cooperative cancellation and status polling.
- See `assets/docs/BACKGROUND_JOBS.md` for details.

### 4.4 Persistence and configuration
- Runtime/process behavior is controlled from `ADSMOD/settings/.env`.
- Database behavior is controlled from `ADSMOD/settings/configurations.json` under `database`:
  - `embedded_database=true`: local SQLite.
  - `embedded_database=false`: external PostgreSQL.

## 5. Frontend Navigation Model

- Top-level sections: `source`, `fitting`, `training`.
- Pages are mounted from `client/src/App.tsx` and switched via `Sidebar`.
- Responsibilities:
  - `ConfigPage`: dataset loading and NIST actions/status.
  - `ModelsPage`: model selection and fit execution.
  - `MachineLearningPage`: dataset build, checkpoint management, and training dashboard.

## 6. Extension Points

- New API capability:
  - add service logic under `server/services`,
  - expose via a router under `server/api`,
  - include the router in `server/app.py`.
- New long-running workflow:
  - execute through `job_manager.start_job(...)`,
  - expose start/status/cancel routes,
  - keep cancellation cooperative.
