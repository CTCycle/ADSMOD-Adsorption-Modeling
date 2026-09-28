# ADS-T4-01 — current Base-profile ML boundary

Date: 2026-09-28
Validated implementation SHA: `2f52eb0fcfbe955418b6b5071d1b5811226d2c2c`
Status: `PASS` for Base-profile gating; positive ML lifecycle remains `BLOCKED`

## Current Base-profile checks

- `app/server/.venv/.adsmod-dependency-state.json` records
  `Development / Base` on Python `3.14.7`.
- The isolated backend became ready and reported `datasets=true`,
  `nist=true`, `fitting=true`, `machine_learning=false`, `training=false`,
  and `checkpoints=false`.
- `/api/v1/system/configuration` returned HTTP 200;
  `/api/v1/training/configuration` and `/api/v1/training/status` returned
  HTTP 404; the generated Base OpenAPI omitted
  `/api/v1/training/configuration`.
- The current Base unit boundary checks passed `11` tests with one intentional
  ML-positive test deselected. The current non-ML architecture/provider subset
  is recorded in [`../ADS-T5-01/summary.md`](../ADS-T5-01/summary.md).

## Remaining Tier 4 gates

The positive `ADS-T4-02` through `ADS-T4-04` lifecycle was not attempted in
the Base environment. The current dependency check found no Torch, RDKit,
Keras, or scikit-learn, so dataset build, real training, checkpoint,
resume, and populated-dashboard claims remain blocked. The exact environment
boundary is recorded in [`../ML-blocker-recheck.md`](../ML-blocker-recheck.md).
