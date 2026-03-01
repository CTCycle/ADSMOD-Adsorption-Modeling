# Background Job Management

ADSMOD uses a centralized background job system to handle long-running operations (fitting, NIST ingestion/enrichment, dataset build, and training) without blocking FastAPI request handling.

## Core Concepts

The system is built around a singleton `JobManager` instance in `ADSMOD/server/services/jobs.py`.

### Execution model
- **Default mode (`thread`)**: Jobs run in daemon threads.
- **Optional mode (`process`)**: Jobs can run in spawned child processes for stronger isolation/cancellation behavior.
- **Cancellation**: Always cooperative; the worker must observe stop signals.

### Job State
Every job is tracked through a thread-safe `JobState`:
- **`job_id`**: A unique 8-character UUID string.
- **`job_type`**: Logical job group (e.g., `fitting`, `nist_fetch`, `training`).
- **`status`**: Current state (`pending`, `running`, `completed`, `failed`, `cancelled`).
- **`progress`**: Float value from 0.0 to 100.0.
- **`result`**: The final output payload (dict) upon successful completion.
- **`error`**: Error message if the job failed.
- **`created_at` / `completed_at`**: Monotonic timestamps for lifecycle tracking.

## Usage Guide

### 1. The Job Manager Singleton
Import the shared instance to interact with the system:
```python
from ADSMOD.server.services.jobs import job_manager
```

### 2. Implementation Pattern
Define a synchronous runner function that performs the heavy work and returns a `dict`.

```python
def my_runner(payload: dict, job_id: str | None = None) -> dict:
    """
    Runs in a worker thread/process.
    """
    if job_id and job_manager.should_stop(job_id):
        return {"status": "cancelled"}
    result = perform_expensive_calculation(payload)
    return {"data": result}
```

### 3. Starting a Job in an API Endpoint
Use `start_job` inside routes/services and expose `job_id` to clients.

```python
@router.post("/start")
def start_processing(payload: Dict):
    if job_manager.is_job_running("MY_JOB_TYPE"):
        raise HTTPException(400, "Job already in progress")

    job_id = job_manager.start_job(
        job_type="MY_JOB_TYPE",
        runner=my_runner,
        args=(payload,)
    )
    return {"job_id": job_id}
```

### 4. Cooperative Cancellation
Cancellation marks the job as cancelled and signals the worker. Your runner must stop cleanly when requested.

## API Interaction

The frontend uses polling with endpoint-specific job URLs:

1. **Start**: domain endpoint returns `JobStartResponse` with `job_id` and optional `poll_interval`.
2. **Poll**:
   - `GET /fitting/jobs/{job_id}`
   - `GET /nist/jobs/{job_id}`
   - `GET /training/jobs/{job_id}`
3. **Cancel**: matching `DELETE` endpoint on the same route.
4. **UI behavior**:
   - The frontend updates progress bars based on the `progress` field.
   - If `status` is `completed`, the frontend displays the `result`.
   - If `status` is `failed`, the frontend displays the `error`.
