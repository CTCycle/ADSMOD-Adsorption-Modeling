# ADSMOD API Surface

Last updated: 2026-08-02

## Core Service Scope

The transitional core service owns non-ML routes only:

- health and root routes
- dataset upload, import preview, metadata, row, and read flows outside training-only workflows
- fitting routes
- NIST and source-collection routes
- canonical user-dataset management routes:
  - `GET /api/datasets`
  - `POST /api/datasets/import/preview`
  - `POST /api/datasets/import/validate`
  - `POST /api/datasets/import/commit`
  - `DELETE /api/datasets/{dataset_id}`
  - `PATCH /api/datasets/{dataset_id}/rename`
  - `PATCH /api/datasets/{dataset_id}/metadata`
  - `GET /api/datasets/{dataset_id}/experiments`
  - `GET /api/datasets/{dataset_id}/experiments/{isotherm_id}/observations`

Dataset metadata is part of the current schema; existing databases must be
recreated. Dataset and experiment selection uses numeric IDs. Fitting accepts a
`dataset_id` and optional `isotherm_id` and resolves the persisted series
server-side.

Core service must not expose `/api/training/*`.
`app/server/app.py` composes the core routes in the unified backend. When
`ADSMOD_ENABLE_ML=true`, it also mounts the ML routes in that same application;
route ownership remains with `ml_service`.

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
Core-only launch paths must not import `ml_service`. The extracted v3 packages
currently expose their separate `/health/*`, `/api/v1/system/capabilities`, and
core snapshot contracts; they do not replace this transitional `/api` surface.
