# ADSMOD User Workflows

Last updated: 2026-08-03

## Main Navigation

The application uses one frontend with these primary routes:

- `datasets` for custom workspace datasets and file import
- `public-data` for NIST-A adsorption experiments
- `public-materials` for NIST adsorbates, adsorbent materials, and existing PubChem enrichment
- `dashboards` for the current dashboard placeholder
- `fitting` for adsorption model fitting
- `training` for processing, training datasets, checkpoints, and the dashboard

Custom dataset management, public adsorption data, and public materials/adsorbates
are standalone top-level routes.

## Upload And Fit A Local Dataset

1. Open `datasets` and import a local file.
2. Upload a `.csv`, `.xls`, or `.xlsx` dataset.
3. Confirm the dataset statistics.
4. Open `fitting`.
5. Select the dataset, model set, optimizer, and iterations.
6. Start fitting and monitor logs.

## Use NIST Data For Fitting

1. Open `public-data`.
2. Run the NIST experiments ping, index, and fetch actions as needed.
3. Confirm status updates.
4. Open `fitting`.
5. Select the resulting workspace dataset.
6. Start fitting and monitor job status.

## Retrieve Public Materials And Adsorbates

1. Open `public-materials`.
2. Use the Adsorbates section for NIST guest-species index and fetch actions.
3. Use the Adsorbent Materials section for NIST host-material index and fetch actions.
4. Run the existing PubChem enrichment action only after records are available locally.
5. Treat NIST retrieval and PubChem enrichment as separate status and provenance steps.

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
