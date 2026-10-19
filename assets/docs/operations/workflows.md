# ADSMOD User Workflows

Last updated: 2026-10-19

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

1. Open `public-data/sources` and set the fraction to acquire for experiments, guests, or hosts.
2. Select the collection's Fetch action. ADSMOD queues a durable run and displays progress, saved, skipped, and failed counts.
3. Cancel between committed batches when needed. Earlier batches remain available.
4. Confirm the imported measurements in the local Adsorption Data view.
5. Open `fitting`, select the NIST workspace dataset, and start fitting.

The index refresh, NIST ping, and enrichment actions remain available for their existing workflows. The overview, source registry, and local record views do not need an external provider to respond.

## Retrieve Public Materials And Adsorbates

1. Open `public-data/overview`.
2. Open `public-data/sources` to review provider registration, the last saved connection state and its freshness, local provenance counts, and recent acquisition activity.
3. Use Refresh status to queue independent provider health checks. The source list remains visible while checks run.
4. Use the Materials, Chemicals, and Structures views to inspect normalized records.
5. Resolve a CID, InChIKey, or name in Chemicals. Search COD by ID, formula, or a narrow text query in Structures, then import one result or select up to 50 COD results for one bounded run.
6. Review progress, counts, retries, and per-record errors in Sources. Local records stay available when a provider is unavailable.

Scheduled NIST refresh is disabled by default. To enable it, configure `application.public_data.auto_sync_enabled`, select one or more `auto_sync_categories`, and choose the interval and fraction in the runtime configuration.

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
