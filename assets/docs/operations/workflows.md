# ADSMOD User Workflows

Last updated: 2026-11-10

## Main Navigation

The application uses one Angular frontend with these primary routes:

- `home` for the default overview of stored data, sources, and modeling (the Sources statistics tiles open Datasets already filtered to provider records)
- `datasets` for datasets, adsorbent materials, and chemical species, and file import
- `datasets/:id` for one dataset (`?experiment=<id>` selects an experiment)
- `help` for the Docs page (tutorials, guides, and tips; `/docs` is reserved by the backend)
- `sources` for the provider workspace (opens Catalog)
- `sources/:view` for Catalog, Acquire, and Structures
- `dashboards/overview` and `dashboards/explore` for cross-dataset data exploration
- `modeling/fitting` for fitting sessions, `modeling/fitting/new` for the new-session wizard, and `modeling/fitting/:id` for one session
- `modeling/models` for the adsorption model catalog and global parameter defaults
- `modeling/deep-learning/:view` for processing, training datasets, checkpoints, and the dashboard (ML profile only)

Datasets manages local data (uploaded and acquired); Sources discovers and
acquires external data. The former `public-data` routes are not supported.

## Upload And Fit A Local Dataset

1. Open `datasets` and use the upload button in the toolbar to import a local file. Rows show experiment, measurement, material, and species counts, a source icon (tooltip names the origin), and icon actions to explore or delete; datasets and materials/species paginate independently (12 per page, up to 25).
2. Upload a `.csv`, `.xls`, or `.xlsx` dataset. In the wizard, review Preview, then Map columns:
   dataset information, data interpretation, scientific units, validation settings, and column mapping.
   Units are searchable; a unit not in the catalog can be saved for future imports or used for the
   current import only. A unit without a conversion definition blocks the import.
   Files with one uptake column per species (for example `uptake_CO2` and `uptake_N2`) are detected
   and listed under Species columns; each species name can be edited. They are converted to one
   measurement row per species, stored under a single experiment. Files that already have one
   adsorbate column are grouped the same way when an experiment contains several species. Species
   columns must share one uptake unit.
3. Validate series: experiments appear as collapsible sections with their observations, warnings, and
   errors. Datasets with blocking errors cannot be saved.
4. After saving, the new dataset is highlighted in the list; Explore dataset opens it directly. Explore
   shows the experiments in a compact navigator (Up/Down/Home/End change the selection; multi-species
   experiments carry a "mix" tag) and their measurements in a worksheet table (or a plot with one
   series per species).
5. To correct data, choose Edit dataset: change dataset metadata, experiment name, material, species,
   temperature and units, or individual measurement values. Save changes validates everything,
   reports field-specific problems, and overwrites the stored values in one transaction; Cancel
   discards the changes. Delete asks for confirmation and states how many experiments, measurements,
   and fitting sessions are removed; materials and species are kept.
6. Open `modeling/fitting` and choose New session.
7. Complete the four steps: Dataset (readiness shows whether every experiment is eligible), Models
   (only enabled, compatible models), Fitting settings (optimizer, evaluations, weighting, optional
   per-session parameter bounds, display units), and Review. Start the session and monitor its log.
8. Open the session from the list to review per-experiment results, the best model, and the
   configuration snapshot. The list filters by name, status, dataset, model, and date; delete a
   session once it is no longer running.

A session fits all experiments of one dataset. To change a model's global bounds or disable it,
use `modeling/models`; bounds are edited in canonical units (Pa, mol/kg) and results are shown in
the session display units (bar, mmol/g by default). Session overrides never change the globals.

## Review Species And Material Identities

The validation step of an upload lists each name in the file and what it will be saved as. Names that cannot be matched are saved as unverified, and uploads never wait on a provider.

Open **Datasets > Identity review** from the notice on the Datasets page, or go to `/datasets/identity-review`. **Find duplicates** proposes likely duplicates. For each item, choose **Keep this record** to merge the other record into it, or **Keep separate** to close the question for good. Back up the SQLite database before approving a batch of merges.

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

1. Open `sources/acquire`. For each NIST collection (adsorbate species, adsorbent materials, adsorption isotherms), use the single Check availability button in the card heading to see stored versus available records (`N / M records`). Availability checks are not acquisitions: they never appear in acquisition history or its counters. Fetch share is a percentage (1 to 100) that only sets how much of the index a fetch requests; once availability is known, each row previews the records it covers.
2. Set the share of the index to consider and select Fetch. Each run is bounded; fetch again to continue where the previous run stopped. Records already stored are never downloaded twice.
3. Follow the run in the Run tracker: progress, saved, already-stored, and failed counts, failed-record reasons, and Cancel between committed batches.
4. Follow "View datasets" to the Datasets page and explore the `NIST ISODB` dataset. NIST experiments can be
   edited like any other; fetching NIST isotherms again asks before overwriting those local edits.
5. To fetch specific records instead, switch Adsorption collections to Targeted: search by compound (name, synonym, or InChIKey), adsorbent material, or experiment (filename or DOI), optionally with a temperature range, tick the isotherms, and select Fetch selected. Already stored isotherms are marked and cannot be selected; fetching also stores the referenced compound and material records. Isotherms whose uptake unit is unsupported are skipped with a reason, as in bulk runs.
6. Open `modeling/fitting`, start a new session, select the NIST ISODB dataset, and start fitting.

The overview, source registry, and local record views do not need an external provider to respond.

## Retrieve Public Materials And Adsorbates

1. Open `sources/acquire`.
2. PubChem: in Single mode search a CID, InChIKey, or name, inspect the matches, and Import one; in Bulk mode set the share and select Enrich stored species to add PubChem identity, synonyms, and properties to stored species that lack them. The card shows how many stored species are enriched (a local ratio; PubChem publishes no total). Fetch 3D structures (Bulk mode) downloads the 3D conformer of enriched species that have none stored yet.
3. COD: in Single mode search by text, formula, or COD ID, select up to 50 results, optionally choose a material to link them to, and import. In Bulk mode set the share and select Fetch structures: stored materials without a structure are looked up by name, and only a unique exact name match is imported and linked: one exact entry, or one exact entry with coordinates among several. Several exact entries with coordinates (polymorphs or phases) are never chosen between; the skip lists their COD IDs so you can link one from Single mode. Unavailable and ambiguous materials are skipped and listed under Jobs → Details (status bar).
4. Each card shows its active or latest run from this visit, with counts and a link to the imported records; Details opens Jobs for failures, Cancel, and Resume. Links from a species or material on the Datasets page open the matching search with it already run, and a failed request keeps the current selection and link target. Each card remembers its Single or Bulk mode.
5. Review species and materials in the Datasets page side column: each opens a detail panel of collapsible sections with identifiers, properties, provenance, a structure preview (2D image and 3D viewer for species; linked COD crystal structures and 3D viewer for materials), and the datasets that use it. Missing data reads "Not available" and inapplicable fields "Not applicable"; a species without a cached conformer can be fetched from its panel. Unreferenced entities can be deleted there. Imported structures are listed in `sources/structures`, where a structure can be linked to or unlinked from a material and its CIF downloaded.
6. Open `sources/catalog` to review providers by data type and saved connection state, and use the refresh icon in the Sources tab bar to queue independent health checks. Local records stay available when a provider is unavailable.

Scheduled NIST refresh is disabled by default. To enable it, configure `application.public_data.auto_sync_enabled`, select one or more `auto_sync_categories`, and choose the interval and fraction in the runtime configuration.

## Build Training Data And Run Training

1. Open the unified UI and navigate to `modeling/deep-learning/processing`.
2. In `Data Processing`, build processed datasets.
3. In `Train datasets`, start a new training run.
4. Use `modeling/deep-learning/dashboard` to monitor progress, metrics, and logs.
5. Training is available only when the optional ML profile is installed and
   capability discovery reports it available.

## Resume From A Checkpoint

1. Open the unified UI `modeling/deep-learning/checkpoints` view.
2. Select a checkpoint.
3. Resume training with additional epochs.
4. Validate resumed metrics in the dashboard.
