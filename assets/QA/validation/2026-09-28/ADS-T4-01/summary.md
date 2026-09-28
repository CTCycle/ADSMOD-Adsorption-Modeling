# ADS-T4-01 — ML profile, capability detection, and Base gating

Date: 2026-09-28
Status: `PASS`

## Profile and topology checks

- The official dependency installer activated the locked
  `Development / ML` profile on Python `3.14.7` with Torch `2.10.0+cu130`,
  Keras `3.13.1`, and scikit-learn `1.8.0`.
- The backend remained the same FastAPI process and reported
  `datasets=true`, `nist=true`, `fitting=true`, `machine_learning=true`,
  `training=true`, and `checkpoints=true`.
- `/api/v1/training/configuration` reported Keras backend `torch`, CUDA
  available, one device, and `NVIDIA GeForce RTX 3060 Laptop GPU`.
- ML-profile OpenAPI generation produced 56 paths and matched the tracked
  canonical snapshot by SHA-256
  `4A7D33EBE0E772FFB868D5C40FF373D0AF68E3460FC842F759F18ADB080428EB`.

## Base-profile boundary

The earlier isolated Base-profile recheck remains valid evidence for the
fail-closed boundary: ML capability flags were false, training routes returned
404, and the Base OpenAPI omitted training paths. The Base and ML profiles use
the same backend topology; the profile selects whether optional ML routes are
registered. No current source imports RDKit, and it is not part of the locked
ML extra, so the former RDKit absence wording was stale and has been removed
from the active blocker description.

The profile gate and the positive lifecycle are now both covered by the dated
`ADS-T4-02`, `ADS-T4-03`, and `ADS-T4-04` evidence.
