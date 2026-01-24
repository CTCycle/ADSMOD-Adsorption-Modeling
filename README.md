# ADSMOD Adsorption Modeling

## 1. Project Overview

ADSMOD is a comprehensive web application designed for the collection, management, and modeling of adsorption data. This project represents the evolution and unification of two predecessor projects: **ADSORFIT** and **NISTADS Adsorption Modeling** (the former name of this repository).

By merging the capabilities of these systems into a single, cohesive platform, ADSMOD provides a robust workflow for researchers and material scientists. The application allows users to:
- **Collect** adsorption isotherms from the NIST Adsorption Database.
- **Enrich** material data with chemical properties fetched from PubChem.
- **Build** curated, standardized datasets suitable for machine learning.
- **Train and Evaluate** deep learning models to predict adsorption behaviors.

The system is organized as a modern web application, featuring a responsive user interface that interacts with a powerful backend for data processing and machine learning tasks. It is designed to abstract away the complexity of raw data handling, allowing users to focus on scientific analysis and model development.

> **Work in Progress**: This project is still under active development. It will be updated regularly, but you may encounter bugs, issues, or incomplete features.



## 2. Model and Dataset

This project utilizes deep learning techniques to model adsorption phenomena.

- **Model**: The core of the learning capabilities is the **SCADS** model architecture. It is designed to learn from complex, sequence-based adsorption data.
- **Learning**: The system relies on **Supervised Learning**, using historical experimental data to train models that can predict adsorption uptake under various conditions.
- **Dataset**:
    - **Primary Source**: Experimental adsorption isotherms are sourced directly from the **NIST Adsorption Database**.
    - **Enrichment**: Chemical properties (e.g., molecular weights, SMILES strings) are sourced from **PubChem**.
    - The application automatically handles the fetching, cleaning, and merging of these distinct data sources to create training-ready datasets.


## 3. Installation

### 3.1 Windows (One Click Setup)

ADSMOD provides an automated installation and launcher script for Windows users, streamlining the setup process.

1.  Navigate to the `ADSMOD` directory.
2.  Locate and run the `start_on_windows.bat` script.

**What this script does:**
- It automatically creates a local Python virtual environment.
- It installs all necessary dependencies (backend and frontend).
- It configures the application environment.
- It launches the application.

**First Run vs. Subsequent Runs:**
- On the **first run**, the script may take some time to download and install all dependencies.
- On **subsequent runs**, it will skip installation and immediately launch the application.

### 3.2 Manual Setup (Advanced)

If you prefer to set up the application manually or are running on a non-Windows environment, ensure you have Python and Node.js installed. You will need to install the backend dependencies from `pyproject.toml` and the frontend dependencies from the `client` directory, then launch the server and client components respectively.


## 4. How to Use

### 4.1 Launching the Application

**Windows:**
Simply double-click `start_on_windows.bat` in the `ADSMOD` folder. This will open a terminal window showing the application logs and typically launch your default web browser to the application's interface (usually `http://localhost:3000` or similar).

### 4.2 Operational Workflow

Once the application is running, the typical workflow involves:

1.  **Data Collection**:
    - Navigate to the "Data" or "Collection" section.
    - Initiate a fetch from the NIST database to populate your local repository with raw isotherms.
    - Enriched material properties are automatically fetched to ensure data completeness.

    ![Database Browser](ADSMOD/assets/figures/04_database_browser.png)

2.  **Dataset Building**:
    - Use the "Processing" tools to convert raw data into structured datasets.
    - Define parameters such as training/validation splits and specific material filters.

    ![Data Configuration](ADSMOD/assets/figures/data_source.png)

3.  **Model Training**:
    - Go to the "Training" area.
    - Select a built dataset and configure training parameters (e.g., batch size, epochs).
    - Start the training job. Real-time progress and metrics (loss, accuracy) will be displayed.

    ![Model Training](ADSMOD/assets/figures/fiitting_page.png)

4.  **Analysis and Inference (to be implemented)**:
    - Review training history and final model performance.
    - Use trained models to predict adsorption isotherms for new or existing material pairings.

    ![Analysis Results](ADSMOD/assets/figures/03_analysis.png)


## 5. Setup and Maintenance
Run `ADSMOD/setup_and_maintenance.bat` to access setup and maintenance actions:

- **Remove logs** - clear `.log` files under `ADSMOD/resources/logs`.
- **Uninstall app** - remove local runtimes and build artifacts (uv, embedded Python, portable Node.js, `node_modules`, `dist`, `.venv`, `uv.lock`) while preserving folder scaffolding.
- **Initialize database** - create or reset the project database schema.


## 6. Resources

The application stores data and artifacts in specific directories, primarily under `ADSMOD/resources`.

- **checkpoints**: Stores trained model weights, training history, and model configuration files.
- **database**: Contains the local SQLite database storing metadata, cached API responses, and experiment indexes.
- **logs**: Application logs for debugging and monitoring background processes.
- **runtimes:** portable Python/uv/Node.js downloaded by the Windows launcher.
- **templates:** starter assets such as the `.env` scaffold



## 7. Configuration
Backend configuration is defined in `ADSMOD/settings/server_configurations.json` and loaded by the API at startup. Runtime overrides and secrets are read from `ADSMOD/settings/.env`. Frontend configuration is read from `ADSMOD/client/.env` during development or build time.

| Variable | Description |
|-|-|
| FASTAPI_HOST | Backend host used by the Windows launcher; defined in `ADSMOD/settings/.env`; default `127.0.0.1`. |
| FASTAPI_PORT | Backend port for uvicorn; defined in `ADSMOD/settings/.env`; default `8000`. |
| RELOAD | Enables uvicorn reload when `true`; defined in `ADSMOD/settings/.env`; default `false`. |
| KERAS_BACKEND | Select torch as backend for Keras. Keep it as default. |



## 8. License

This project is licensed under the **MIT License**. See `LICENSE` for full terms.