# Background Job Management

Last updated: 2026-04-08

ADSMOD runs long operations in background workers to avoid blocking API requests.

## 1. Core Runtime

- Central manager: `ADSMOD/server/services/jobs.py` (`job_manager` singleton).
- Job state model: `ADSMOD/server/domain/jobs.py` (`JobState`, response schemas).
- Primary workloads:
  - fitting jobs,
  - NIST fetch/index/fetch/enrich jobs,
  - training dataset build jobs,
  - training execution/resume jobs.

## 2. Execution and Cancellation Model

- `start_job(...)` supports:
  - `run_mode="thread"` (default),
  - `run_mode="process"` when process isolation is needed.
- Jobs transition through:
  - `pending` -> `running` -> `completed|failed|cancelled`.
- Cancellation is cooperative:
  - `cancel_job(job_id)` sets stop intent,
  - runners should check `job_manager.should_stop(job_id)` where applicable.

## 3. Tracked State Contract

Each job tracks:
- `job_id` (8-char id),
- `job_type` (logical family),
- `status`,
- `progress` (0-100),
- `result` (dict payload),
- `error` (string on failure),
- `created_at` and `completed_at` (monotonic timestamps).

## 4. API Contract Pattern

Every long-running API flow should expose:

1. start endpoint returning `JobStartResponse`,
2. status endpoint returning `JobStatusResponse`,
3. list endpoint returning `JobListResponse` where useful,
4. cancel endpoint for cooperative stop.

In frontend usage, call the `/api/...` endpoints for same-origin routing.

## 5. Active Endpoint Families

- Fitting:
  - `POST /api/fitting/run`
  - `GET /api/fitting/jobs`
  - `GET /api/fitting/jobs/{job_id}`
  - `DELETE /api/fitting/jobs/{job_id}`

- NIST:
  - `POST /api/nist/fetch`
  - `POST /api/nist/properties`
  - `POST /api/nist/categories/{category}/index`
  - `POST /api/nist/categories/{category}/fetch`
  - `POST /api/nist/categories/{category}/enrich`
  - `GET /api/nist/jobs`
  - `GET /api/nist/jobs/{job_id}`
  - `DELETE /api/nist/jobs/{job_id}`

- Training dataset build and training runs:
  - `POST /api/training/build-dataset`
  - `GET /api/training/jobs`
  - `GET /api/training/jobs/{job_id}`
  - `DELETE /api/training/jobs/{job_id}`
  - `POST /api/training/start`
  - `POST /api/training/resume`
  - `POST /api/training/stop`
  - `GET /api/training/status`

## 6. Implementation Guardrails

- Keep runners side-effect aware and idempotent where possible.
- Normalize error messages for UI-safe display.
- Always emit progress/result updates through the manager to keep polling payloads stable.
