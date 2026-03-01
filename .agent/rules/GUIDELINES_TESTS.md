# HOW TO TEST

This document describes the testing strategy and execution workflow for the current ADSMOD webapp.

## Overview

ADSMOD tests are Python-based and currently include:
- End-to-end tests (Playwright + pytest) for UI/API flows.
- Unit and server tests for backend logic and persistence behavior.
- Backend performance checks for heavier training/data paths.

## Test Suite Structure

```text
tests/
|-- run_tests.bat
|-- conftest.py
|-- fixtures/
|   `-- sample_adsorption.csv
|-- e2e/
|   |-- test_app_flow.py
|   |-- test_datasets_api.py
|   |-- test_fitting_api.py
|   |-- test_nist_api.py
|   `-- test_training_api.py
|-- unit/
|   `-- ...
|-- server/
|   `-- ...
`-- backend/
    `-- performance/
```

## Quick Start (Windows)

Run:

```cmd
tests\run_tests.bat
```

The runner will:
1. Validate runtime prerequisites from the existing `.venv`.
2. Start backend if it is not already running.
3. Build and serve frontend if needed.
4. Execute `pytest tests`.
5. Stop only the servers it started.

## Prerequisites

- Python 3.14+ environment available in `.venv`.
- Optional test dependencies installed when required:
  - `pytest`
  - `pytest-playwright`
  - `psutil`
- Playwright browsers installed:

```cmd
.\.venv\Scripts\python.exe -m playwright install
```

> [!TIP]
> For Windows setup, run `ADSMOD\start_on_windows.bat` with `OPTIONAL_DEPENDENCIES=true` in `ADSMOD/settings/.env`.

## Manual Run

1. Start application services (or run them separately):
   - `ADSMOD\start_on_windows.bat`
2. Run tests:

```cmd
.\.venv\Scripts\python.exe -m pytest tests -v
```

## URL and Environment Resolution

`tests/conftest.py` resolves URLs from:
- `ADSMOD/settings/.env` (`FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, `UI_PORT`)
- Optional overrides:
  - `ADSMOD_TEST_FRONTEND_URL`
  - `ADSMOD_TEST_BACKEND_URL`

Wildcard bind hosts (`0.0.0.0`, `::`) are normalized to `127.0.0.1` for client requests.

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `base_url` | session | Frontend URL |
| `api_base_url` | session | Backend base URL |
| `api_context` | session | Playwright `APIRequestContext` |
| `page` | function | Fresh browser page |
| `sample_csv_path` | session | CSV fixture path |

## API Endpoints for New E2E Coverage

Use only active router prefixes:
- `/datasets`
- `/fitting`
- `/nist`
- `/training`

### Dataset endpoints
- `POST /datasets/load`
- `GET /datasets/names`
- `GET /datasets/by-name/{dataset_name}`

### Fitting endpoints
- `POST /fitting/run`
- `GET /fitting/nist-dataset`
- `GET /fitting/jobs`
- `GET /fitting/jobs/{job_id}`
- `DELETE /fitting/jobs/{job_id}`

### NIST endpoints
- `GET /nist/status`
- `POST /nist/fetch`
- `POST /nist/properties`
- `GET /nist/categories/status`
- `POST /nist/categories/{category}/ping`
- `POST /nist/categories/{category}/index`
- `POST /nist/categories/{category}/fetch`
- `POST /nist/categories/{category}/enrich`
- `GET /nist/jobs`
- `GET /nist/jobs/{job_id}`
- `DELETE /nist/jobs/{job_id}`

### Training endpoints
- `GET /training/datasets`
- `GET /training/dataset-sources`
- `DELETE /training/dataset-source`
- `POST /training/build-dataset`
- `GET /training/processed-datasets`
- `GET /training/dataset-info`
- `DELETE /training/dataset`
- `GET /training/jobs`
- `GET /training/jobs/{job_id}`
- `DELETE /training/jobs/{job_id}`
- `GET /training/checkpoints`
- `GET /training/checkpoints/{checkpoint_name}`
- `DELETE /training/checkpoints/{checkpoint_name}`
- `POST /training/start`
- `POST /training/resume`
- `POST /training/stop`
- `GET /training/status`

## Important Notes

- Avoid adding new tests for `/browser/...` endpoints unless backend routes are restored and wired in `ADSMOD/server/app.py`.
- Keep tests isolated and deterministic:
  - Use Arrange-Act-Assert.
  - Avoid depending on mutable external services unless explicitly marked.
  - Use minimal payloads for NIST/training tests to limit runtime.

## Troubleshooting

- **Connection refused**: Ensure backend/frontend URLs match `ADSMOD/settings/.env`.
- **Missing Playwright**: run `.\.venv\Scripts\python.exe -m playwright install`.
- **Missing pytest deps**: enable optional dependencies and rerun launcher.
- **Port conflicts**: free the configured `FASTAPI_PORT`/`UI_PORT` or update `.env`.
