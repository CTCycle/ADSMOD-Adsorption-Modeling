# ADSMOD API Surface

Last updated: 2026-06-03

## Core Service Scope

Core service owns non-ML routes only:

- health and root routes
- dataset upload and read flows outside training-only workflows
- fitting routes
- NIST and source-collection routes

Core service must not expose `/api/training/*`.

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
