# Legacy repository to merge: `legacy/NISTADS`

This document summarizes the architecture of the legacy “NISTADS” application living under `legacy/`. This is a **desktop GUI** (PySide6) for:

- collecting adsorption data from the NIST adsorption database (via their API),
- enriching it with external chemical properties (PubChem),
- building curated datasets, and
- training / evaluating / running inference with a deep learning model (SCADS).

It is intended to be fully merged into the main ADSMOD webapp over time.

---

## 1) Repository map (legacy app)

Root of legacy project:

- `legacy/NISTADS/app/app.py` — application entry point (PySide6 `QApplication`)
- `legacy/NISTADS/app/client/` — GUI layer: main window wiring, dialogs, event handlers, workers
- `legacy/NISTADS/app/layout/main_window.ui` — Qt Designer UI layout
- `legacy/NISTADS/app/utils/` — application services, ML stack, persistence, configuration
- `legacy/NISTADS/resources/` — mutable state: database files, checkpoints, logs, templates
- `legacy/NISTADS/setup/` — environment setup (`setup/.env`, scripts)
- `legacy/NISTADS/start_on_windows.bat` / `setup_and_maintenance.bat` — launcher + maintenance scripts

Core packages:

- `legacy/NISTADS/app/client/window.py` — main UI controller (binds widgets → actions → workers)
- `legacy/NISTADS/app/client/events.py` — “service façade” used by the UI (dataset/model/validation pipelines)
- `legacy/NISTADS/app/client/workers.py` — background execution primitives (thread and process workers)
- `legacy/NISTADS/app/utils/repository/database.py` — SQLAlchemy DB + table definitions + export helpers
- `legacy/NISTADS/app/utils/repository/serializer.py` — serialization for datasets/checkpoints/models
- `legacy/NISTADS/app/utils/services/*` — dataset building, preprocessing, data fetching, conversions, loaders
- `legacy/NISTADS/app/utils/learning/*` — Keras/TensorFlow model definitions, training, inference, metrics
- `legacy/NISTADS/app/utils/validation/*` — evaluation utilities for dataset and model quality

---

## 2) Runtime architecture (desktop)

### 2.1 Entry point

`legacy/NISTADS/app/app.py`:

- creates a `QApplication`
- applies a Qt material theme (`apply_style(...)`)
- instantiates `MainWindow(UI_PATH)` and enters the Qt event loop

### 2.2 Threading and long-running tasks

The GUI is designed to stay responsive while running heavy workloads:

- I/O-heavy jobs (API calls, dataset processing) run via `ThreadWorker` (Qt thread pool)
- CPU/GPU-heavy jobs (training/inference) typically run via `ProcessWorker` (multiprocessing)

Workers emit Qt signals:

- `progress` (percent updates)
- `finished` (result payload)
- `error` (exception + traceback)
- `interrupted` (user-requested cancel)

The GUI owns the worker lifecycle and offers “Stop” functionality via an interruption flag/event.

Key implementation: `legacy/NISTADS/app/client/workers.py`.

---

## 3) Layering and patterns (legacy)

The legacy app follows a pragmatic “GUI + event handlers + services” structure, mixing several recognizable patterns:

- **MV*/Controller style**: `MainWindow` acts as a controller that binds view widgets to commands.
- **Service Layer**: `DatasetEvents`, `ModelEvents`, `ValidationEvents` expose high-level workflows used by UI callbacks.
- **Repository pattern**: `NISTADSDatabase` + SQLAlchemy models, exposed as a singleton-like `database`.
- **Pipeline pattern**: dataset building and ML training are implemented as sequential pipelines calling smaller service objects.
- **Strategy/Factory**:
  - model selection uses a map `MODEL_COMPONENTS` that picks model class + dataloader based on config
  - device selection logic lives in `DeviceConfig`

---

## 4) Key subsystems

### 4.1 GUI subsystem (`app/client`)

#### `window.py` (MainWindow)

`MainWindow` is the central coordinator:

- Loads Qt UI (`main_window.ui`) at runtime via `QUiLoader`
- Reads current configuration via `Configuration`
- Initializes the database (`database.initialize_database()`)
- Creates “persistent handlers”:
  - `GraphicsHandler` — renders Matplotlib figures to Qt pixmaps, loads images
  - `DatasetEvents`, `ValidationEvents`, `ModelEvents` — domain workflows
- Binds UI actions/buttons to methods that:
  - validate current UI state,
  - instantiate a worker with a pipeline function,
  - connect signals to completion/error handlers,
  - start the worker (thread/process).

#### `events.py` (workflows)

`DatasetEvents`, `ModelEvents`, `ValidationEvents` host the actual long workflows. They typically:

- load/save tables through `DataSerializer` / `database`
- call into `utils/services/*` for preprocessing/building datasets
- call into `utils/learning/*` for training/inference/evaluation
- frequently check worker interruption state (cooperative cancellation)

#### `dialogs.py`

Small Qt dialogs for saving/loading configuration JSON presets (stored under `resources/configurations/`).

### 4.2 Configuration and environment

- Default in-memory configuration dictionary is defined in `legacy/NISTADS/app/utils/configuration.py::Configuration`
- Environment variables are loaded from `legacy/NISTADS/setup/.env` through `legacy/NISTADS/app/utils/variables.py`

Notable: configuration is largely “flat” (single dictionary), and differs from ADSMOD’s dataclass-based server settings.

### 4.3 Data fetching and dataset composition (`utils/services`)

#### Remote APIs

`legacy/NISTADS/app/utils/services/server.py`:

- Uses the NIST adsorption API:
  - experiments list: `https://adsorption.nist.gov/isodb/api/isotherms.json`
  - per-experiment JSON: `.../isotherm/{id}.json` (fetched concurrently)
  - materials endpoints:
    - gas data: `.../gas/{InChIKey}.json`
    - adsorbent materials: `.../material/{hashkey}.json`
- Uses `aiohttp` + an `asyncio.Semaphore` to limit concurrency (`parallel_tasks`)
- Integrates with GUI cancellation:
  - checks interrupt flag during fetch loops
  - cancels pending asyncio tasks when interrupted

`legacy/NISTADS/app/utils/services/properties.py`:

- Uses `pubchempy` to query PubChem for molecular properties (molecular weight, formula, SMILES)
- Normalizes naming and merges fetched properties into existing tables

#### Dataset building pipeline components

`legacy/NISTADS/app/utils/services/builder.py`:

- Transforms raw JSON payloads (nested fields) into flat tabular datasets
- Splits “single component” vs “binary mixture” datasets
- Explodes sequence fields so the database stores one row per measurement for raw datasets

`legacy/NISTADS/app/utils/services/conversion.py`:

- Normalizes pressure units (e.g., bar → Pa)
- Normalizes uptake units into a common basis (mmol/g), with several supported unit conversions

`legacy/NISTADS/app/utils/services/sequences.py`:

- Cleans pressure/uptake series:
  - filters by min/max measurement count
  - trims leading zeros, pads/truncates sequences to fixed length (Keras `pad_sequences`)
- Tokenizes and encodes SMILES strings using a custom tokenizer and a generated vocabulary

`legacy/NISTADS/app/utils/services/sanitizer.py`:

- Aggregates measurement rows into per-experiment rows
- Filters out-of-bounds values
- Encodes adsorbents as categorical classes (LabelEncoder-like mapping)
- Normalizes scalar features + sequence columns (pressure/uptake)
- Splits train/validation using stratified splitting over (adsorbate, adsorbent) combinations

`legacy/NISTADS/app/utils/services/loader.py`:

- Builds ML dataloaders for training and inference
- Important dependency mismatch with current guidelines:
  - this module imports `tensorflow as tf` and uses `tf.data` (legacy stack)

### 4.4 Persistence (`utils/repository`)

`legacy/NISTADS/app/utils/repository/database.py`:

- SQLAlchemy declarative models for:
  - raw adsorption measurement tables (single component / binary mixture)
  - materials tables (adsorbates, adsorbents)
  - training dataset table
  - predicted adsorption table
  - checkpoint summary tables
- Implements:
  - `save_into_database` (destructive overwrite)
  - `upsert_into_database` (SQLite insert-on-conflict-update)
  - `export_all_tables_as_csv`
  - `delete_all_data`

`legacy/NISTADS/app/utils/repository/serializer.py`:

- `DataSerializer` for:
  - loading datasets and metadata
  - serializing/deserializing sequence columns between list and string formats
- `ModelSerializer` for:
  - checkpoint folder creation under `resources/checkpoints/`
  - saving/loading Keras models (`.keras`)
  - saving training configuration, metadata, and history JSON
  - loading checkpoints with custom objects (masked metrics, schedulers)

### 4.5 ML stack (`utils/learning`)

The legacy ML implementation is “SCADS” (transformer-ish) and includes:

- custom embeddings/encoders/transformers (`utils/learning/models/*`)
- training routines (`utils/learning/training/*`)
- inference predictor (`utils/learning/inference/predictor.py`)
- custom metrics and callbacks (`utils/learning/metrics.py`, `utils/learning/callbacks.py`)

Framework coupling:

- Keras + TensorFlow are used directly in several places (training and `tf.data` input pipelines).

---

## 5) What overlaps with ADSMOD today

There are strong conceptual overlaps with ADSMOD’s backend:

- A “pipeline” shape:
  - ingest → preprocess → persist → compute → persist → visualize
- A persistence façade that maps DataFrames to SQL tables
- A configuration+environment bootstrap with `.env`
- A “models registry” concept (legacy: ML model selection; ADSMOD: isotherm equation set)

However, the legacy codebase is **GUI-driven**, whereas ADSMOD is already **API-driven**, which is a better fit for merging.

---

## 6) Recommended merge strategy (pragmatic plan)

The least risky merge is to move *capabilities* (pipelines) into ADSMOD’s backend as API services, and rebuild UI in React incrementally.

### Phase A — Extract backend services (no UI changes yet)

- Port legacy “data fetch + preprocessing + dataset build” to ADSMOD backend:
  - new service package (e.g., `ADSMOD/server/utils/services/nistads_*`)
  - new routers (e.g., `/nistads/*`) for:
    - fetch experiments index
    - fetch experiments payloads (async)
    - enrich with PubChem properties
    - build datasets and persist them
- Use ADSMOD’s existing configuration approach (JSON + `.env`) to avoid UI-only config state.
- Keep persistence in SQLite initially (embedded), with separate tables or separate SQLite DB file.

### Phase B — Introduce background jobs

Legacy relies on threads/processes; the webapp needs a job system:

- Add a lightweight job queue:
  - simplest: FastAPI background tasks + local process pool, with job IDs stored in SQLite
  - scalable: Celery/RQ + Redis (only if you accept the additional moving parts)
- Expose:
  - `POST /jobs/...` start a job
  - `GET /jobs/{id}` query status/progress
  - `GET /jobs/{id}/logs` stream logs (optional)

### Phase C — Migrate ML training/inference

This is the most complex part due to compute + dependencies:

- Decide if SCADS stays TensorFlow-based or is migrated to pure Keras 3 (torch backend) to align with modern Keras usage.
- Separate concerns:
  - dataset generation (CPU) vs training (GPU) vs inference (CPU/GPU)
- Add model registry and checkpoint management to ADSMOD resources:
  - unify with `ADSMOD/resources/` layout
  - store checkpoint metadata and training history similarly to legacy

### Phase D — Rebuild UI in React

Map legacy tabs to web pages:

- Data tab → “NISTADS Data” page (fetch + preprocess + explore)
- Model tab → “Training / Inference” pages (start job + progress + results)
- Viewer tab → “Artifacts” page (plots, reports, saved checkpoints)

---

## 7) Integration risks and “gotchas”

- **Dependency mismatch**: legacy uses `tensorflow` in `loader.py`; ADSMOD currently depends on scientific stack but not TF.
- **Long-running tasks**: training/inference must not block FastAPI workers; it needs job isolation.
- **Data volume**: fetching and storing large experiment datasets can produce large SQLite files; consider chunking and streaming APIs.
- **Schema drift**: legacy stores sequences as space-separated strings in some places; ADSMOD stores sequences as JSON strings for processed datasets. Choose one encoding and standardize.
- **Config drift**: legacy uses a flat config dict; ADSMOD uses structured server settings. A compatibility layer may be needed during migration.

