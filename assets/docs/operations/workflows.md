# ADSMOD User Workflows

Last updated: 2026-11-05

## Main Navigation

The application uses one Angular frontend with these primary routes:

- `datasets` for datasets, adsorbent materials, and chemical species, and file import
- `datasets/:id` for one dataset (`?experiment=<id>` selects an experiment)
- `help` for the Docs page (tutorials, guides, and tips; `/docs` is reserved by the backend)
- `sources` for the provider workspace (opens Catalog)
- `sources/:view` for Catalog, Acquire, and Structures
- `dashboards/overview` and `dashboards/explore` for cross-dataset data exploration
- `fitting` for adsorption model fitting
- `training` for processing, training datasets, checkpoints, and the dashboard

Datasets manages local data (uploaded and acquired); Sources discovers and
acquires external data. The former `public-data` routes are not supported.

## Upload And Fit A Local Dataset

1. Open `datasets` and use the upload button in the toolbar to import a local file. Rows show experiment, measurement, material, and species counts, a source icon (tooltip names the origin), and icon actions to explore or delete; datasets and materials/species paginate independently (12 per page, up to 25).
2. Upload a `.csv`, `.xls`, or `.xlsx` dataset. In the wizard, review Preview, then Map columns:
   dataset information, data interpretation, scientific units, validation settings, and column mapping.
   Units are searchable; a unit not in the catalog can be saved for future imports or used for the
   current import only. A unit without a conversion definition blocks the import.
3. Validate series: experiments appear as collapsible sections with their observations, warnings, and
   errors. Datasets with blocking errors cannot be saved.
4. After saving, the new dataset is highlighted in the list; Explore dataset opens it directly. Explore
   shows the experiments in a navigator and their measurements in a worksheet table (or a plot).
5. To correct data, choose Edit dataset: change dataset metadata, experiment name, material, species,
   temperature and units, or individual measurement values. Save changes validates everything,
   reports field-specific problems, and overwrites the stored values in one transaction; Cancel
   discards the changes. Delete asks for confirmation and states how many experiments, measurements,
   and fitting runs are removed; materials and species are kept.
6. Open `fitting`.
7. Select the dataset (search experiments when a dataset has many), model set, optimizer, and iterations.
8. Start fitting and monitor logs.

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

1. Open `sources/acquire`. For each NIST collection (adsorbate species, adsorbent materials, adsorption isotherms), use Check availability to see stored versus available records (`N / M records`). Availability checks are not acquisitions: they never appear in acquisition history or its counters. Fetch share only sets how much of the index a fetch requests.
2. Set the share of the index to consider and select Fetch. Each run is bounded; fetch again to continue where the previous run stopped. Records already stored are never downloaded twice.
3. Follow the run in the Run tracker: progress, saved, already-stored, and failed counts, failed-record reasons, and Cancel between committed batches.
4. Follow "View datasets" to the Datasets page and explore the `NIST ISODB` dataset. NIST experiments can be
   edited like any other; fetching NIST isotherms again asks before overwriting those local edits.
5. Open `fitting`, select the NIST ISODB dataset, and start fitting.

The overview, source registry, and local record views do not need an external provider to respond.

## Retrieve Public Materials And Adsorbates

1. Open `sources/acquire`.
2. PubChem: resolve a CID, InChIKey, or name, or select Enrich stored chemicals to add PubChem identity, synonyms, and properties to stored species that lack them.
3. COD: search by text, formula, or COD ID, select up to 50 results, optionally choose a material to link them to, and import.
4. Follow the Run tracker and open the imported records from its links.
5. Review species and materials in the Datasets page side column: each opens a detail panel with identifiers, properties, linked structures, and the datasets that use it. Unreferenced entities can be deleted there. Imported structures are listed in `sources/structures`, where a structure can be linked to or unlinked from a material and its CIF downloaded.
6. Open `sources/catalog` to review providers by data type and saved connection state, and use the refresh icon in the Sources tab bar to queue independent health checks. Local records stay available when a provider is unavailable.

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
