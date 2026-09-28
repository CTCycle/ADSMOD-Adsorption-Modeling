# ADS-T4-03 — real training, status, metrics, and cancellation

Date: 2026-09-28
Status: `PASS`
Runtime: Development / ML profile with CUDA available

## Positive run

- `/api/v1/training/configuration` reported Keras backend `torch`, one CUDA
  device, and `NVIDIA GeForce RTX 3060 Laptop GPU`.
- Training session/job `20b2253c` ran the processed dataset with the
  `SCADS Series` model for one epoch and completed successfully.
- Final status reported `is_training=false`, epoch `1/1`, progress `100%`,
  loss `0.39273571968078613`, validation loss `0.14401447772979736`, masked
  R2 `-3.7881507873535156`, and validation masked R2
  `-1.131443977355957`.
- The status history and training log contained the completed epoch and
  terminal `Training completed.` event.

## Cancellation

- Cancellation session/job `016cc087` was started for a 100-epoch run and
  observed active at `0/100` before the stop request.
- `POST /api/v1/training/stop` returned `status=stopped` and
  `Training stop requested.`; the job then reached terminal `cancelled`.
- After the asynchronous finalization callback completed, the live training
  status reported `is_training=false`, zero progress, an empty metrics/history
  payload, and the log entries `Stop requested by user...` and
  `Training cancelled.`

The brief interval where the job was cancelled but the status endpoint still
reported active was rechecked after the callback settled; no stuck-training
condition remained.

## Remaining limitation

The fixture and one-epoch run certify execution and lifecycle contracts only.
They do not claim model quality, convergence, or production-scale training
readiness.
