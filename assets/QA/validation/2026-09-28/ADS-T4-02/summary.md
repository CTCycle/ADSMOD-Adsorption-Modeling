# ADS-T4-02 — processed training dataset and immutable snapshot

Date: 2026-09-28
Status: `PASS`
Runtime: Development / ML profile, Python 3.14.7, Torch 2.10.0+cu130,
Keras 3.13.1, scikit-learn 1.8.0

## Validation evidence

- The valid-SMILES fixture was imported through the current API. Preview
  returned the expected SHA-256
  `441ba11a14713d5d769d6c8c704366b2fd7d5fd09a6b280cd9dd9d9187718a07`,
  eight rows, two experiments, and no validation issues.
- Validation and commit succeeded for dataset label
  `qa-current-20260928` (two experiments, eight observations).
- Build job `b040d1ce` completed successfully with two total samples, one
  train sample, and one validation sample. The persisted processed dataset
  hash was
  `2282ba5912f545b3f0daccc0d0637954ef568497ad22328820e3efe592d17138`.
- The current training configuration exposed the expected dataset metadata,
  including the CO2 SMILES sequence and molecular-weight feature.

## Fixes found during validation

- `AggregateDatasets.aggregate_adsorption_measurements` now retains
  `adsorbate_SMILE` and `adsorbate_molecular_weight`, which are required by the
  training builder after uploaded adsorption rows are grouped by experiment.
- `SnapshotStore.create` now reuses an existing immutable snapshot with the
  same content hash. This makes a retry after a later build-stage failure
  idempotent instead of raising a unique-constraint error.
- Regression coverage was added for both behaviors. The focused combined
  backend selection passed `40` tests.

## Remaining limitation

This is a tiny synthetic fixture used to certify the application lifecycle.
It establishes dataset construction and persistence behavior, not scientific
model quality or representative production-data performance.
