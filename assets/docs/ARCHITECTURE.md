# ADSMOD Architecture

**ADSMOD** is a local, browser-based tool for adsorption isotherm fitting and training. It combines a Python FastAPI backend with a React/Vite frontend.

---

## 1. Repository Structure

- **`ADSMOD/`**: Main application source.
  - **`server/`**: Python backend (FastAPI, SQLAlchemy, SciPy, PyTorch/Keras).
    - `routes/`: API routers (`datasets`, `fitting`, `nist`, `training`).
    - `services/`: Domain logic (`data/`, `modeling/`, `training.py`, `jobs.py`).
    - `repositories/`: DB backends, query layer, and serializers.
    - `learning/`: Training runtime, model definitions, and callbacks.
  - **`client/`**: React + TypeScript UI.
    - `src/pages/`: Main pages (`ConfigPage`, `ModelsPage`, `MachineLearningPage`).
    - `src/services/`: API clients and polling helpers.
  - **`settings/`**: Runtime environment and static configuration.
  - **`resources/`**: Data-only folder (database, checkpoints, logs).
  - **`runtimes/`**: Portable Python/uv/Node.js managed by launcher scripts.
  - **`start_on_windows.bat`**: One-click Windows setup + launch.
- **`tests/`**: End-to-end, unit, server, and backend performance tests.

---

## 2. System Overview

### Processes
- **Backend**: Uvicorn running FastAPI on `FASTAPI_HOST:FASTAPI_PORT` from `ADSMOD/settings/.env`.
- **Frontend (local launcher)**: Vite preview server (`npm run preview`) on `UI_HOST:UI_PORT` from `ADSMOD/settings/.env`.
- **Frontend (packaged desktop mode)**: FastAPI serves static SPA assets from `client/dist` when Tauri mode is enabled.
- **Communication**: Frontend uses same-origin `/api` proxying in both local webapp and packaged desktop modes, avoiding CORS configuration overhead.

### Backend Layers
1. **Routes**: Validation and HTTP handling.
2. **Services**: Business logic (dataset ingestion, NIST workflows, fitting, training orchestration).
3. **Repositories**: Data persistence through backend abstraction (SQLite or PostgreSQL).
4. **Learning**: ML-specific logic (model runtime, training loops, checkpoint handling).

---

## 3. Key Subsystems

### 3.1 Data Pipeline
1. **Ingestion**: Upload CSV/Excel datasets via `/datasets/load` into `adsorption_data`.
2. **NIST acquisition**: Pull experiments/guest/host datasets via `/nist/...` endpoints.
3. **Processing**: `DatasetBuilder` and `DatasetCompositionService` prepare curated training datasets.
4. **Training**: Training jobs consume processed datasets and persist checkpoints with metadata.
5. **Compatibility**: Resume/selection flows compare checkpoint metadata with available dataset hashes.

### 3.2 Job Management
Long-running tasks (fitting, NIST ingestion/enrichment, dataset build, training) run through a centralized **JobManager** in `ADSMOD/server/services/jobs.py`. Default execution mode is thread-based, with optional process-based mode for isolation.

> **Detailed Documentation**: See [BACKGROUND_JOBS.md](./BACKGROUND_JOBS.md) for implementation details, threading models, and code patterns.

### 3.3 Persistence
- **Database modes**: Embedded SQLite (`DB_EMBEDDED=true`) or external PostgreSQL (`DB_EMBEDDED=false`).
- **Tables**: Raw adsorption data, NIST datasets, processed training data, and fitting outputs.
- **Metadata**: Dataset and checkpoint metadata are stored for reproducibility and resume safety.
- **Sequence storage**: Structured list-like fields are persisted through JSON-compatible serialization.
- **Pagination**: Repository backends support `limit` and `offset` for large-table reads.
- **Query centralization**: SQL/ORM query construction is centralized under `ADSMOD/server/repositories/queries/`; services, routes, and non-query repository modules consume those query helpers.
- **ORM-first reads/writes**: Application-level table reads and row counts use SQLAlchemy ORM mapped models (`select(...)`, `func.count(...)`) rather than raw SQL strings; raw SQL remains only for DB bootstrap/session configuration paths (for example PostgreSQL database creation and encoding checks, SQLite PRAGMA).

### 3.4 Logging
- **Application logger**: Backend modules use `ADSMOD.server.common.utils.logger.logger` (logger name: `ADSMOD`).
- **Level control**: Runtime verbosity is controlled through `logger.level_of_log` (default `DEBUG`).
- **Handler rules**:
  - Console handler remains `INFO` with minimal format.
  - File handler remains `DEBUG` with timestamped detailed format.
- **Scope filtering**: Output handlers are filtered to the `ADSMOD` logger namespace only.
- **Framework noise suppression**: `uvicorn`, `uvicorn.error`, `uvicorn.access`, and `fastapi` loggers are forced to `CRITICAL` with propagation disabled.
- **Launcher behavior**: Windows launcher and test runner start Uvicorn with `--no-access-log --log-level critical` to avoid HTTP/access log spam.

---

## 4. Extending ADSMOD

- **New model**: Update backend model logic in `ADSMOD/server/services/modeling/models.py` and frontend configuration in `ADSMOD/client/src/adsorptionModels.ts`.
- **New API capability**: Add logic under `ADSMOD/server/services/`, expose via a router in `ADSMOD/server/routes/`, and wire it in `ADSMOD/server/app.py`.
- **New long-running workflow**: Use `job_manager.start_job(...)` from `ADSMOD/server/services/jobs.py`, provide polling endpoints, and keep cancellation cooperative.

### 3.5 Frontend Navigation and Layout (2026 refresh)
- Global navigation is tab-based in the top header (`source`, `fitting`, `training`) and is wired in `ADSMOD/client/src/App.tsx` + `src/components/Sidebar.tsx`.
- `ConfigPage` (`source`) uses a two-column structure, each column organized as: description row, widget row, and markdown/log row.
- `ModelsPage` (`fitting`) uses a wider fitting panel and a 4-column model grid on large viewports, degrading responsively on smaller screens.
- `MachineLearningPage` (`training`) owns a page-local left toolbar with four internal views:
  - `Data Processing` (dataset composition widget)
  - `Training datasets` (dataset selection/manage widget)
  - `checkpoints` (resume/manage widget)
  - `Training dashboard` (metrics/charts/log widget)
- Archived training dataset labels (`archived::...`) are filtered from the frontend dataset list view to avoid showing non-actionable historical entries.

