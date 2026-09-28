# ADS-T4-04 — checkpoint, resume, deletion, and populated training dashboard

Date: 2026-09-28
Status: `PASS` for the training-dashboard scope; top-level dashboard remains
`PARTIAL` under `ISSUE-005`

## Checkpoint lifecycle

- The successful T4-03 run created checkpoint
  `qa-t4-20260928_20260928T095842`.
- Checkpoint inspection confirmed the dataset hash
  `2282ba5912f545b3f0daccc0d0637954ef568497ad22328820e3efe592d17138`,
  compatible model/config metadata, two samples split one-to-one between
  train and validation, the SMILES vocabulary, and normalization statistics.
- Resume session/job `417340b8` accepted one additional epoch and completed
  successfully. The checkpoint history grew from one to two epochs; final
  status was `2/2` with progress `100%` and final validation loss
  `0.14331848919391632`.
- Deleting the checkpoint returned success and the checkpoint listing became
  empty.

## Rendered dashboard evidence

The Codex in-app Browser rendered `/training/dashboard` after the resumed run.
The observed page showed `Training Dashboard`, `Run status: Idle`, epoch `2/2`,
the populated train/validation loss and R2 cards/charts, progress `100%`, the
training log, and `Backend Online`. This is direct rendered evidence for the
populated training-dashboard route.

## Remaining limitation

The separate top-level `/dashboards` route is still a product placeholder and
was not changed by this validation slice. `ui.dashboards` therefore remains
`PARTIAL` and `ISSUE-005` remains open for product-scope implementation and
validation. The positive training-dashboard result no longer depends on the
ML blocker.
