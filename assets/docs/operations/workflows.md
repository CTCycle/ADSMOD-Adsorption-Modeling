# ADSMOD User Workflows

Last updated: 2026-10-21

## Main Navigation

The application uses one Angular frontend with these primary routes:

- `datasets` for custom workspace datasets and file import
- `public-data` for the public-data workspace (opens Overview)
- `public-data/:view` for Overview, Acquire, Adsorption Data, Materials, Chemicals, Structures, and Sources
- `dashboards/overview` and `dashboards/explore` for cross-dataset data exploration
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

## Explore And Compare Stored Data

1. Open `dashboards/overview` to review database composition, temperature
   coverage, the most measured materials and adsorbates, and spans per pressure
   basis. Select a ranked material or adsorbate to open Explore filtered to it.
2. In `dashboards/explore`, filter by source, pressure basis, temperature,
   material, adsorbate, or dataset. Facet counts update with each filter.
3. Optionally select isotherms in the results table; otherwise charts use all
   matching isotherms up to the series limit.
4. Choose an analysis: isotherm curves (colour by temperature, material,
   adsorbate, or dataset; linear or log pressure), uptake at one pressure
   against temperature, or a distribution histogram.
5. Select a curve or row to inspect provenance, material, and adsorbate
   details. The address bar keeps the view, so it can be bookmarked; it is
   recomputed from current data when reopened.

## Use NIST Data For Fitting

1. Open `public-data/acquire`. For each NIST collection (adsorbate species, adsorbent materials, adsorption isotherms), use Check availability to see stored versus available records.
2. Set the share of the index to consider and select Fetch. Each run is bounded; fetch again to continue where the previous run stopped. Records already stored are never downloaded twice.
3. Follow the run in the Run tracker: progress, saved, already-stored, and failed counts, failed-record reasons, and Cancel between committed batches.
4. Follow "View isotherms →" to the Adsorption Data view, then open a record to reach its material and adsorbates.
5. Open `fitting`, select the NIST workspace dataset, and start fitting.

The overview, source registry, and local record views do not need an external provider to respond.

## Retrieve Public Materials And Adsorbates

1. Open `public-data/acquire`.
2. PubChem: resolve a CID, InChIKey, or name, or select Enrich stored chemicals to add PubChem identity, synonyms, and properties to stored chemicals that lack them. From a chemical's detail, Resolve on PubChem attaches the result to that chemical instead of creating a new one.
3. COD: search by text, formula, or COD ID, select up to 50 results, optionally choose a material to link them to, and import. From a material's detail, Find COD structures opens this panel with the material preselected.
4. Follow the Run tracker and open the imported records from its links.
5. Browse Materials, Chemicals, Structures, and Adsorption Data. Records link to each other (isotherm → material and adsorbates, material → structures and isotherms, chemical → isotherms), every detail has a shareable `?id=` URL, and a structure can be linked to or unlinked from a material and its CIF downloaded.
6. Open `public-data/sources` to review provider registration and saved connection state, and use Refresh status to queue independent health checks. Local records stay available when a provider is unavailable.

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
