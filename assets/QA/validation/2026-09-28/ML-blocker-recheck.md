# Current ML validation evidence

Date: 2026-09-28

Status: `PASS` for the bounded ML installation and workflow scope. This note
retains the hardware/profile boundary that cannot be reconstructed from the
repository tests alone; current status is summarized in
[`assets/docs/project_status_ledger.md`](../../../docs/project_status_ledger.md).

## Profile boundary

The Base profile correctly reported ML unavailable, omitted training routes, and
kept the normal application usable. The locked Development / ML profile used
Python 3.14.7 with Torch 2.10.0+cu130, Keras 3.13.1, scikit-learn 1.8.0,
the Torch Keras backend, and one NVIDIA GeForce RTX 3060 Laptop GPU. The
unified FastAPI process registered the training and checkpoint routes, and the
ML-enabled OpenAPI contract contained 56 paths.

## Positive lifecycle

The live recheck built a valid-SMILES processed dataset and immutable snapshot,
completed a one-epoch training run, reached terminal cancellation for a longer
run, created and resumed a compatible checkpoint, deleted it, and rendered the
populated `/training/dashboard` route. The focused regression selection passed
40 tests.

The run certified application lifecycle and persistence behavior only. Its
tiny synthetic fixture does not establish model quality, convergence,
representative-data behavior, long-duration stability, or release readiness.
The separate top-level `/dashboards` placeholder remains `PARTIAL` under
`ISSUE-005`; it is not an ML dependency blocker.
