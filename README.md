# ADSMOD Adsorption Modeling

[![Release](https://img.shields.io/github/v/release/CTCycle/ADSMOD-Adsorption-Modeling?display_name=tag)](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/releases)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-22.12.0-5FA04E?logo=node.js&logoColor=white)](https://nodejs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/workflows/ci.yml?query=branch%3Adevelop)

Last updated: 2026-08-31

## Purpose

ADSMOD is a local web application for organizing adsorption data and turning it
into a consistent, inspectable analysis workflow. It brings together local
measurements, public reference data, adsorption-model fitting, and optional
SCADS machine-learning workflows in one workspace.

It is useful when you want to:

- import and review adsorption datasets from common spreadsheet formats;
- enrich adsorbates and adsorbent materials with public NIST and PubChem data;
- standardize measurements before comparing experiments;
- fit several adsorption models and compare their reported metrics; and
- prepare training data, run machine-learning experiments, and resume them from
  saved checkpoints when the ML service is enabled.

ADSMOD is designed to keep the normal workflow close to the data. Imported
datasets and analysis results are stored in local application storage by
default, while public-data actions retrieve information from their respective
external sources when requested.

## How the application works

An adsorption isotherm describes how much of a substance is held by a material
as equilibrium pressure changes under stated conditions, usually at a fixed
temperature. The measured points form the evidence; a model is a mathematical
curve used to summarize and compare that evidence.

ADSMOD guides the user through four connected stages:

1. **Collect** measurements from local files or public NIST collections.
2. **Prepare** the data by reviewing column roles, grouping experiments, and
   confirming units and material or adsorbate information.
3. **Fit** one or more model curves to a selected experiment and inspect the
   resulting metrics and status information.
4. **Learn** from prepared datasets through the optional machine-learning
   workspace.

Different adsorption models represent different assumptions. For example,
Langmuir models describe saturation on one or more site families, while
Freundlich describes heterogeneous behaviour without an explicit saturation
limit. Other available models provide flexible heterogeneous, coverage-based,
micropore-filling, or exponential-saturation descriptions. The fitting page
shows each model's equation and assumptions so that you can choose deliberately.

A numerical fit is a comparison aid, not proof that a model is the physical
mechanism behind an experiment. Units, pressure basis, temperature, data
quality, and the scientific context should always be considered alongside the
reported metrics. AICc is available as a model-comparison aid because it
balances fit quality with model complexity; it should not be treated as a
substitute for scientific judgment.

At a high level, ADSMOD uses an Angular browser interface, Python application
services, a local SQLite database by default, and an optional machine-learning
service. The Windows launcher supplies and coordinates the local Python and
Node.js runtimes needed by the application.

## Windows setup

The supported end-user workflow is the Windows local web launcher. Start it
from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1
```

When the menu appears, choose **Launch Application**. The launcher prepares the
local runtime when needed, installs or repairs application dependencies, builds
the web interface, starts the required local services, waits for them to be
ready, and opens the application in your browser.

The first launch can take longer because the launcher may need to download
portable runtimes and packages. An internet connection is therefore helpful on
first launch and whenever dependencies or public data must be retrieved.

If the browser does not open automatically, use the local address printed by
the launcher. The application runs on your computer; it is not a hosted cloud
workspace.

### macOS and Linux

The official one-command launcher documented by this project is currently for
Windows. macOS and Linux users may be able to run the services manually, but a
supported one-command workflow for those platforms is not currently provided.
The advanced startup notes are available in
[`assets/docs/runtime/startup.md`](assets/docs/runtime/startup.md).

## Manual setup

Manual dependency installation and service startup are intended for project
maintainers or advanced users who are developing the application. They are not
needed for normal use: the Windows launcher is the intended entry point and
handles the required setup for you.

If you do need the advanced workflow, use the project runtime documentation so
that all services use the same application settings and data location:

- [`assets/docs/runtime/startup.md`](assets/docs/runtime/startup.md)
- [`assets/docs/runtime/modes.md`](assets/docs/runtime/modes.md)
- [`assets/docs/runtime/configuration.md`](assets/docs/runtime/configuration.md)

## Runtime behavior and data

ADSMOD can run in a standard analysis mode or in an ML-capable mode. The
standard mode includes the dataset, public-data, public-materials, and fitting
workflows. Machine-learning training requires the optional ML service to be
enabled; if it is not enabled, training may correctly appear as unavailable.

The default local setup uses an embedded SQLite database. Local application
storage contains the workspace database, logs, generated machine-learning
artifacts, and saved checkpoints. The launcher manages the runtime settings and
normally removes the need for users to edit configuration files by hand. On
Windows, the default storage is under the ADSMOD folder in your local
application-data directory.

## Main workflows

### Import and prepare a local dataset

1. Open **Custom Datasets**.
2. Import a `.csv`, `.xls`, or `.xlsx` file.
3. Review the import preview. ADSMOD suggests likely roles for columns such as
   pressure, uptake, temperature, and experiment identifiers.
4. Confirm or correct the column mapping, units, pressure basis, and experiment
   grouping. If the file represents one experiment, explicitly confirm that
   choice; if it contains several experiments, select the columns that separate
   them.
5. Save the dataset and review its summary before fitting.

The import step is deliberately reviewable. Clear headers, explicit units,
numeric measurement values, and a clear way to distinguish experiments make
the results more reliable. Compatible units are standardized for analysis while
the source information remains available for review.

### Use public NIST and PubChem data

Open **Public Data** to work with NIST adsorption experiments. Use the available
status, index, and fetch actions as needed, then wait for the activity updates
to finish before using the resulting workspace data.

Open **Public Materials** to retrieve reference information for:

- **Adsorbates**, the guest species involved in adsorption; and
- **Adsorbent Materials**, the host materials that provide the adsorption
  surface or pore structure.

NIST retrieval and PubChem enrichment are separate steps with separate status
and provenance. Run the PubChem enrichment action only after the relevant
records have been retrieved locally. These actions require access to the public
source services and may take time for larger collections.

### Fit adsorption models

1. Open **Fitting**.
2. Select a dataset and its experiment or isotherm.
3. Choose the weighting, optimization method, and maximum iteration budget.
4. Enable the models you want to compare. Expand a model card to review its
   equation, assumptions, and parameter controls.
5. Select **Start Fitting** and monitor the fitting log.
6. Review the completed status and metrics together with the original data and
   experimental conditions.

The fitting process estimates model parameters that make each selected curve
agree as closely as possible with the measured points under the chosen fitting
rules. It does not automatically decide which model is scientifically correct.
Completed fits and their metrics remain available in the local workspace for
later review.

### Prepare data and run machine learning

Machine-learning workflows are available when the ML-capable runtime is
enabled:

1. Open **Training** and use **Data Processing** to build processed datasets.
2. In **Train datasets**, start a new training run using the processed data.
3. Use **Training Dashboard** to follow progress, metrics, and logs.
4. Open **Checkpoints** to inspect saved states and resume a run with additional
   training when appropriate.

Training can take substantially longer than importing or fitting, depending on
the dataset, model, and available hardware. The status views are the best
indication of whether a long-running operation is still progressing.

### Dashboards

The **Dashboards** area provides a workspace for dashboard views as the
application grows. The primary analysis workflow currently lives in Custom
Datasets, Public Data, Public Materials, Fitting, and Training.

Representative UI views:

![Dataset workspace](assets/figures/home.png)

![Fitting workspace](assets/figures/fitting.png)

![Training workspace](assets/figures/training-datasets.png)

## Important expectations and limitations

- The quality of a fit depends on the quality, units, grouping, and conditions
  represented in the source data.
- Public-data and enrichment actions depend on external NIST or PubChem
  availability and an active internet connection.
- Fitting, collection, and training are status-based, potentially long-running
  operations. Allow the activity or fitting log to update before assuming an
  action has failed.
- Training is optional and is not available in the standard analysis mode.
- The documented supported deployment is a local Windows workspace. It does
  not provide cloud synchronization or a hosted multi-user service by default.

## Troubleshooting

### PowerShell will not run the launcher

Run the launcher from the repository root with the documented command, including
the temporary execution-policy bypass:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1
```

If Windows security software blocks the script or downloaded runtime, approve
the repository and its downloaded components only if they come from a trusted
project source, or ask your administrator for help.

### The first launch is slow or dependency setup fails

The first launch may be downloading runtimes and packages. Check the internet
connection, allow the process to finish, and try again. If the launcher menu is
available, choose **Install / Update Dependencies**, then choose
**Launch Application** again.

### The browser does not open or the application is unreachable

Wait for the launcher to report that the services are ready, then open the
local address printed in the launcher window. If an earlier ADSMOD session is
still open, close it and launch again. Also check whether security software is
blocking local browser access. The launcher will report a clear error if a
required service cannot become ready.

### A dataset import is rejected or the preview is unclear

Use a `.csv`, `.xls`, or `.xlsx` file with a header row and numeric measurement
columns. Avoid blank or duplicate headers. In the import wizard, map pressure
and uptake explicitly, confirm units, and choose the correct grouping for one or
multiple experiments. Do not accept a suggested mapping without reviewing it.

### NIST or PubChem retrieval returns no records

Confirm that the computer can reach the public source, then retry the relevant
action and read the status updates. NIST retrieval and PubChem enrichment are
independent, so a successful NIST step does not mean PubChem enrichment has also
run.

### Training is unavailable

Training requires the optional ML-capable runtime. The standard analysis setup
intentionally reports training as unavailable; dataset import and fitting can
still be used normally. Ask the person who prepared the installation to enable
the ML service, then relaunch the application.

### Database initialization fails

For a new or empty installation, choose **Initialize Database** from the
launcher menu and try again after closing any running ADSMOD session. If the
database already contains valuable work, do not delete it or try to repair it by
guessing. Make a backup and consult the detailed project guidance or a project
maintainer, especially if the error mentions an unknown or incomplete schema.

### A job appears to be stuck

Check the activity or fitting log and allow additional time for large imports,
public collections, fits, or training runs. If there is no progress, relaunch
the application and retry the operation. Preserve the displayed error and the
time of the failure if you need to report the problem.

### I need to clear temporary files or reset local data

Use the launcher menu carefully:

- **Clear Cache** removes disposable runtime and test-tool caches.
- **Remove Logs** removes generated log files.
- **Remove Checkpoints** deletes saved training checkpoints.
- **Remove All Data** deletes the local database, uploaded dataset records,
  checkpoints, and generated logs while preserving the application files.

The last two actions can remove work that is difficult to recover. Export or
back up anything important first.

## Testing and validation

Normal users do not need to run the project checks. Project maintainers can use
**Run Test Suite** from the launcher menu after installing the development
dependencies. The launcher also provides **Rebuild Frontend** when the local web
interface needs to be rebuilt.

## Resources and maintenance

Use the launcher menu for the supported maintenance actions:

- initialize the local database;
- install or update dependencies;
- rebuild the web interface;
- check for and apply application updates;
- remove logs or disposable caches;
- remove saved training checkpoints; and
- uninstall local runtimes and build artifacts.

The update action is designed to avoid overwriting local changes. If the local
application copy has unsaved repository changes, resolve or preserve them
before attempting an update.

More detailed project documentation is available from
[`assets/docs/project_index.md`](assets/docs/project_index.md), including
advanced runtime, architecture, UI, operations, and troubleshooting notes.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
