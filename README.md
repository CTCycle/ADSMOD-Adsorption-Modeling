# ADSMOD Adsorption Modeling

<p align="center"><img src='app/client/public/adsmod-logo.png' alt='ADSMOD logo' width="112"></p>

**A local workspace for preparing adsorption datasets, retrieving reference data, fitting models, and exploring optional machine-learning workflows.**


[![License](https://img.shields.io/github/license/CTCycle/ADSMOD-Adsorption-Modeling?style=flat-square)](LICENSE) [![Release](https://img.shields.io/github/v/release/CTCycle/ADSMOD-Adsorption-Modeling?display_name=tag&style=flat-square)](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/releases) [![CI](https://img.shields.io/github/actions/workflow/status/CTCycle/ADSMOD-Adsorption-Modeling/ci.yml?branch=develop&label=CI&style=flat-square)](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/workflows/ci.yml?query=branch%3Adevelop) [![Python](https://img.shields.io/badge/Python-3.14.7-3776AB?logo=python&logoColor=white&style=flat-square)](https://www.python.org/) [![Angular](https://img.shields.io/badge/Angular-21-DD0031?logo=angular&logoColor=white&style=flat-square)](https://angular.dev/) [![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?logo=windows&logoColor=white&style=flat-square)](https://www.microsoft.com/windows)

## Overview

ADSMOD is a Windows-first web application for organizing adsorption data and building an inspectable analysis workflow. It combines local dataset import, public reference data, adsorption-model fitting, and an optional machine-learning workspace. The application runs locally and uses SQLite by default; PostgreSQL is also supported.

Public-data retrieval depends on the availability of external sources such as NIST, PubChem, and COD. Numerical fitting helps compare candidate models but does not establish a physical mechanism.

## Key Features

- Import CSV, XLS, and XLSX adsorption datasets, map their columns, validate units and experiment groups, and retain saved datasets.
- Retrieve and organize public adsorption, material, chemical, and structure data with source provenance.
- Compare enabled adsorption models through asynchronous fitting runs, with metrics, logs, cancellation, and saved results.
- Use the optional ML profile for data preparation, training datasets, checkpoints, and training dashboards.
- Keep the local data and analysis workflow in one browser-based workspace.

## Screenshots

![Dataset workspace](assets/figures/home.png)

![Fitting workspace](assets/figures/fitting.png)

## Technology Stack

- Python 3.14.7 and FastAPI backend.
- Angular 21 web client, built with Node.js 22.13.0.
- SQLite by default, with PostgreSQL support.
- The Windows launcher provisions portable Python, uv, and Node.js runtimes.

## Installation and Usage

From the repository root in PowerShell, start the launcher:

```powershell
.\start_on_windows.ps1
```

Choose **Launch Application**. On first use, the launcher may download runtimes and packages before opening the local application. The default profile does not enable ML routes; select the ML profile when setting up the optional training workflow.

## Configuration

The launcher reads `data/adsmod.json` by default. Use `-DataPath` or `ADSMOD_DATA_DIR` to select another configuration directory. The JSON configuration controls runtime settings and the storage root for the database, logs, checkpoints, and optional ML artifacts.

## Documentation

- [Runtime startup](assets/docs/runtime/startup.md)
- [Runtime configuration](assets/docs/runtime/configuration.md)

## License

Licensed under the MIT License. See [LICENSE](LICENSE).
