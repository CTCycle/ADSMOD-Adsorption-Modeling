# ADSMOD User Workflows

Last updated: 2026-08-02

## Main Navigation

The application uses one frontend with these primary routes:

- `datasets` for workspace datasets, file import, and NIST-A collection
- `dashboards` for the current dashboard placeholder
- `fitting` for adsorption model fitting
- `training` for processing, training datasets, checkpoints, and the dashboard

The source-management and NIST screens are embedded within `datasets`; they are
not standalone top-level routes.

## Upload And Fit A Local Dataset

1. Open `datasets`.
2. Upload a `.csv`, `.xls`, or `.xlsx` dataset.
3. Confirm the dataset statistics.
4. Open `fitting`.
5. Select the dataset, model set, optimizer, and iterations.
6. Start fitting and monitor logs.

## Use NIST Data For Fitting

1. Open `datasets`.
2. Run NIST category actions such as ping, index, fetch, or enrich as needed.
3. Confirm status updates.
4. Open `fitting`.
5. Select the resulting workspace dataset.
6. Start fitting and monitor job status.

## Build Training Data And Run Training

1. Open the unified UI and navigate to `training`.
2. In `Data Processing`, build processed datasets.
3. In `Train datasets`, start a new training run.
4. Use `Training Dashboard` to monitor progress, metrics, and logs.

## Resume From A Checkpoint

1. Open the unified UI `training/checkpoints` view.
2. Select a checkpoint.
3. Resume training with additional epochs.
4. Validate resumed metrics in the dashboard.
