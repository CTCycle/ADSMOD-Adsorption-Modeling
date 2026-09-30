# ADSMOD User Workflows

Last updated: 2026-09-30

## Main Navigation

The application uses one Angular frontend with these primary routes:

- `datasets` for custom workspace datasets and file import
- `public-data` for NIST-A adsorption experiments
- `public-data/:view` for NIST adsorption data, materials, chemicals, structures, sources, and PubChem/COD enrichment
- `dashboards` for the current dashboard placeholder
- `fitting` for adsorption model fitting
- `training` for processing, training datasets, checkpoints, and the dashboard

Custom dataset management and public data are standalone workspaces. The former
split `public-materials` destination is not a supported route.

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

1. Open `public-data/overview`.
2. Use `public-data/sources` for NIST guest-species and host-material index/fetch actions.
3. Use the Materials, Chemicals, and Structures views to inspect normalized records.
4. Run PubChem or COD enrichment/import only after the relevant local records are available.
5. Treat each provider action as a separate status and provenance step.

## Build Training Data And Run Training

1. Open the unified UI and navigate to `training/processing`.
2. In `Data Processing`, build processed datasets.
3. In `Train datasets`, start a new training run.
4. Use `training/dashboard` to monitor progress, metrics, and logs.
5. Training is available only when the optional ML profile is installed and
   capability discovery reports it available.

## Resume From A Checkpoint

1. Open the unified UI `training/checkpoints` view.
2. Select a checkpoint.
3. Resume training with additional epochs.
4. Validate resumed metrics in the dashboard.
