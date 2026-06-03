# ADSMOD User Workflows

Last updated: 2026-06-03

## Main Navigation

The application is split into two frontends:

- Core UI in `app/client` for `source` and `fitting`
- ML UI in `app/ml_client` for `training`

## Upload And Fit A Local Dataset

1. Open `source`.
2. Upload a `.csv`, `.xls`, or `.xlsx` dataset.
3. Confirm the dataset statistics.
4. Open `fitting`.
5. Select the dataset, model set, optimizer, and iterations.
6. Start fitting and monitor logs.

## Use NIST Data For Fitting

1. Open `source`.
2. Run NIST category actions such as ping, index, fetch, or enrich as needed.
3. Confirm status updates.
4. Open `fitting`.
5. Select `NIST-A Collection`.
6. Start fitting and monitor job status.

## Build Training Data And Run Training

1. Open the ML UI in `app/ml_client`.
2. In `Data Processing`, build processed datasets.
3. In `Train datasets`, start a new training run.
4. Use `Training Dashboard` to monitor progress, metrics, and logs.

## Resume From A Checkpoint

1. Open the ML UI `Checkpoints` view.
2. Select a checkpoint.
3. Resume training with additional epochs.
4. Validate resumed metrics in the dashboard.
