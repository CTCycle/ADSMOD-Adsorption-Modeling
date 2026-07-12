# ADSMOD API Surface

Last updated: 2026-07-11

## Core Service Scope

Core service owns non-ML routes only:

- health and root routes
- dataset upload and read flows outside training-only workflows
- fitting routes
- NIST and source-collection routes
- canonical user-dataset management routes:
  - `POST /api/datasets`, `GET /api/datasets`
  - `DELETE /api/datasets/by-name/{dataset_name}`
  - `PATCH /api/datasets/by-name/{dataset_name}/rename`
  - `GET`/`PATCH /api/datasets/by-name/{dataset_name}/metadata`
  - `GET`/`PATCH /api/datasets/by-name/{dataset_name}/rows`

Dataset metadata is part of the fresh schema; existing databases must be recreated. Legacy dataset load, names, full-dataset retrieval, and NIST fitting-export endpoints are not exposed. Fitting resolves `{ source: "uploaded", dataset_name }` or `{ source: "nist" }` server-side.

Core service must not expose `/api/training/*`.
`app/server/app.py` may compose those routes into the unified backend only when `ADSMOD_ENABLE_ML=true`; route ownership remains with `ml_service`.

## ML Service Scope

ML service owns training workflows:

- `/api/training/datasets`
- `/api/training/dataset-sources`
- `/api/training/dataset-source`
- `/api/training/build-dataset`
- `/api/training/processed-datasets`
- `/api/training/dataset-info`
- `/api/training/dataset`
- `/api/training/jobs`
- `/api/training/jobs/{job_id}`
- `/api/training/checkpoints`
- `/api/training/checkpoints/{checkpoint_name}`
- `/api/training/start`
- `/api/training/resume`
- `/api/training/stop`
- `/api/training/status`

Training routes belong only to `ml_service`, even when they are mounted by the unified backend entrypoint.
Core-only launch paths must not import `ml_service`.
