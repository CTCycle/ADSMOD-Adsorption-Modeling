<p align="center"><img src="app/client/public/adsmod-logo.png" alt="ADSMOD logo" width="80"></p>

<h1 align="center">ADSMOD</h1>

<p align="center">Prepare adsorption datasets, fit isotherm models, and explore results in a local web workspace.</p>

<p align="center">
  <a href="#workflows">Workflows</a> ·
  <a href="#start-adsmod">Start ADSMOD</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#development">Development</a> ·
  <a href="assets/docs/project_index.md">Documentation</a>
</p>

<p align="center"><img src="assets/figures/fitting-results.jpg" alt="Completed fit comparing Langmuir, Sips, and Freundlich models for one sample isotherm" width="760"></p>

<p align="center"><em>A completed fit for one 14-point experiment.</em></p>

## Overview

ADSMOD is a local web application for organizing adsorption measurements and analyzing isotherms. Import your own datasets, retrieve reference records from public providers, compare model fits, and explore stored measurements. SQLite is the default database; PostgreSQL is also supported. Model training is available with the optional ML profile.

## Workflows

### Prepare a dataset

Import a CSV, XLS, or XLSX file, map its columns, and review the detected experiment groups and validation result. Saved datasets remain available to fitting and dashboard workflows.

<p align="center"><img src="assets/figures/custom-datasets.jpg" alt="Custom dataset detail showing source file validation and detected column mappings" width="760"></p>

<p align="center"><em>Dataset details include the source file, validation status, and saved column mapping.</em></p>

### Retrieve public data

The Public Data workspace can retrieve adsorption data from NIST, resolve chemical records with PubChem, and search for structures in the Crystallography Open Database (COD). Imported records retain source provenance. Provider requests need network access, and availability depends on the source.

### Fit adsorption models

Choose a dataset and experiment, then select the candidate models, weighting, optimizer, and iteration limit. A run shows each model's status and reports RMSE, R², and corrected Akaike information criterion (AICc), alongside the fitting log.

A numerical fit compares candidate functions against the observations; it does not establish a physical adsorption mechanism.

### Explore stored measurements

Dashboards summarize the stored isotherms and let you filter by source, pressure basis, temperature, material, adsorbate, or dataset. Explore views can plot isotherm curves, uptake against temperature at a selected pressure, or a value distribution.

### Train models (optional)

The optional ML profile adds data processing, training datasets, checkpoints, and a training dashboard. It is separate from the base installation.

## Start ADSMOD

On Windows, install PowerShell 7 and run the launcher from the repository root:

```powershell
git clone https://github.com/CTCycle/ADSMOD-Adsorption-Modeling.git
cd ADSMOD-Adsorption-Modeling
.\start_on_windows.ps1
```

Choose **Launch Application**. On first use, the launcher provisions the pinned portable runtimes and dependencies, then builds the web client when needed. This can take several minutes if downloads are required. Keep the launcher console open while using ADSMOD; press **Ctrl+C** there to stop it.

The launcher supports base and ML dependency profiles. Select the ML profile during setup to use the training workflow.

## Configuration

ADSMOD stores its local database, logs, and training artifacts under `%LOCALAPPDATA%\ADSMOD` by default. Set `ADSMOD_STORAGE_ROOT` in `settings/.env` or in the process environment to use another directory.

SQLite is selected by default. PostgreSQL connection settings are managed on **Settings > Database** and saved in `database.json` under the storage root. Host, port, and CORS settings are documented in [runtime configuration](assets/docs/runtime/configuration.md).

## Development

The backend uses Python 3.14+ and FastAPI; the client uses Angular 21 and TypeScript. In normal use, the FastAPI process also serves the built Angular client. For frontend work, `npm run dev` starts the Angular development server. The [runtime startup guide](assets/docs/runtime/startup.md) covers environment setup and manual startup.

See the [user workflows](assets/docs/operations/workflows.md) for detailed procedures, or start from the [project documentation index](assets/docs/project_index.md).

## Contributing

Issues and pull requests are welcome. Describe the problem or change and keep contributions focused.

## License

[MIT](LICENSE)
