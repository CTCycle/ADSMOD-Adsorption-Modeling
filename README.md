# NISTADS: NIST/ARPA-E dataset composer and modeling

## 1. Introduction
The NISTADS project delves into the fascinating world of adsorption, a surface-based process where a film of particles (adsorbate) accumulate on the surface of a material (adsorbent). This phenomenon plays a pivotal role in numerous industries. For instance, it is widely applied in water treatment facilities for the purification of water, in air filters to improve air quality, and in the automotive industry within catalytic converters to reduce harmful emissions. The adsorption of compounds is usually quantified by measuring the adsorption isotherm a given adsorbate/adsorbent combination. The objective of this project is two-fold: 1) to collect adsorption isotherms data from the NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials (https://adsorption.nist.gov/index.php#home) through their dedicated API; 2) build a machine learning model that can accurately predict the adsorbed amount of a specific guest-host combination under various conditions, by leveraging the data from the NIST/ARPA-E Database.

This could have significant implications for industries that rely on these materials, potentially leading to more efficient processes and better materials design. As such, this project takes a different approach compared to fitting adsorption data with theoretical model for adsorption constants calculation, instead proposing the use of a deep learning approach to understand adsorption isotherm patterns by leveraging a large volume of experimental data.

![Adsorbent material](NISTADS/assets/5A_with_gas.png)  

## 2. Adsorption datasets
Users can collect data on adsorbent materials and adsorbate species, along with adsorption isotherm experiments. The data is retrieved asynchronously to enhance processing speed. Conveniently, the app will split adsorption isotherm data in two different datasets (Single Component ADSorption and Binary Mixture ADSorption). Since NISTADS is focused on predicting single component adsorption isotherms, it will make use of the single component dataset (SCADS) for the model training. As the NIST-ARPA-E database does not provide chemical information for either adsorbate species or adsorbent materials, these details are gathered from external sources. For adsorbate species, NISTADS utilizes the PUG REST API (see PubChemPy documentation for more details) to enrich the dataset with molecular properties such as molecular weight and canonical SMILES. However, obtaining information on adsorbent materials is more challenging, as no publicly available API offers this data. 

**Data preprocessing:** The single-component adsorption dataset is processed through a custom pipeline. Initially, experiments containing negative values for temperature, pressure, or uptake are removed, along with any measurements falling outside predefined pressure and uptake boundaries (as specified in the configuration). Next, pressure and uptake values are standardized to consistent units — Pascals for pressure and mol/g for uptake. After this refinement, the dataset is enriched with molecular properties, including molecular weight and SMILES representations for both adsorbate species and adsorbent materials. The pressure and uptake series are then normalized using predefined upper bounds as ceilings. Finally, all sequences are reshaped to a uniform length via post-padding and re-normalized to ensure consistency across the dataset.

## 3. Machine learning model
SCADS (Single Component ADSorption) is a deep learning model specifically developed to predict adsorption isotherms for porous materials using a variety of molecular and experimental descriptors. At its core, SCADS leverages the flexibility and power of transformer-based neural architectures to learn complex, physically meaningful relationships between chemical structures, material classes, experimental state variables, and pressure conditions.

SCADS takes as input a detailed set of features describing the adsorption system, including the adsorbate SMILE sequence and molecular weight, the categorically-encoded adsorbents, the experiment temperature and the pressure series over which the adsorption isotherm is to be predicted. Each of these components is embedded into a shared, high-dimensional space using learnable neural embeddings. For sequential inputs such as SMILES, positional encodings are also added to preserve the order and meaning of the sequence. Masking mechanisms are built in to ensure that padding values do not contaminate the learned representations.

Once embedded, these features are processed through a series of transformer encoder layers, using multi-head self-attention to capture intricate dependencies within and between the different molecular and contextual descriptors. In parallel, a dedicated state encoder transforms experimental state variables, such as temperature, into dense vectors that the model can use to modulate its predictions.

A unique aspect of SCADS is its pressure series encoder, which applies cross-attention between the input pressure series and the context-rich molecular representation produced by the previous layers. This design enables the model to dynamically adapt its predictions to changing pressure conditions, which is essential for accurately modeling adsorption isotherms across a broad range of experimental scenarios. The final decoder head, known as the Q Decoder, combines the encoded pressure and state information and transforms them through a series of dense layers. Temperature and other state variables are incorporated via scaling mechanisms to ensure that predictions remain physically plausible (higher temperature would correspond to lower uptake)

## 4. Installation
The project targets Windows 10/11 and requires roughly 2 GB of free disk space for the embedded Python runtime, dependencies, checkpoints, and datasets. A CUDA-capable NVIDIA GPU is recommended but not mandatory. Ensure you have the latest GPU drivers installed when enabling TorchInductor + Triton acceleration.

1. **Download the project**: clone the repository or extract the release archive into a writable location (avoid paths that require admin privileges).
2. **Configure environment variables**: copy `NISTADS/resources/templates/.env` into `NISTADS/setup/.env` and adjust values (e.g., backend selection).
3. **Run `start_on_windows.bat`**: the bootstrapper installs a portable Python 3.12 build, downloads Astral’s `uv`, syncs dependencies from `pyproject.toml`, prunes caches, then launches the UI through `uv run`. The script is idempotent—rerun it any time to repair the environment or re-open the app.

Running the script the first time can take several minutes depending on bandwidth. Subsequent runs reuse the cached Python runtime and only re-sync packages when `pyproject.toml` changes.

### 4.1 Just-In-Time (JIT) Compiler
`torch.compile` is enabled throughout the training and inference pipelines. TorchInductor optimizes the computation graph, performs kernel fusion, and lowers operations to Triton-generated kernels on NVIDIA GPUs or to optimized CPU kernels otherwise. Triton is bundled automatically so no separate CUDA toolkit installation is required.

### 4.2 Manual or developer installation
If you prefer managing Python yourself (for debugging or CI):

1. Install Python 3.12.x and `uv` (https://github.com/astral-sh/uv).
2. From the repository root run `uv sync` to create a virtual environment with the versions pinned in `pyproject.toml`.
3. Copy `.env` as described earlier and ensure the `KERAS_BACKEND` is set to `torch`.
4. Launch the UI with `uv run python NISTADS/app/app.py`.
The installation process for Windows is fully automated. Simply run the script `start_on_windows.bat` to begin. During its initial execution, the script installs portable Python, necessary dependencies, minimizing user interaction and ensuring all components are ready for local use.  


## 5. How to use
Launch the application by double-clicking `start_on_windows.bat` (or via `uv run python NISTADS/app/app.py`). On startup the UI loads the last-used configuration, scans the resources folder, and initializes worker pools so long-running jobs (training, inference, validation) do not block the interface.

1. **Prepare data**: verify that `resources/database/images` (training) and `resources/database/inference` (inference) contain the expected files. 
2. **Adjust configuration**: use the toolbar to load/save configuration templates or modify each parameter manually from the UI.
3. **Run a pipeline**: pick an action under the Data, Model, or Viewer tabs. Progress bars, log panes, and popup notifications keep you informed. Background workers can be interrupted at any time.

**Data tab:** dataset analysis and validation.

- Extract data from the NIST database and organize it into a structured format.
- Data retrieval is performed concurrently via the NIST/ARPA-E Database API, enabling fast access by maximizing parallel HTTP requests. 
- Additional molecular properties of adsorbates are fetched using the Pug REST API and integrated into the main database. 
- Use Dynamic Time Warping (DTW) on pressure series to clusterize adsorption isotherms based on curve shapes

Eventually, it is possible to build the training dataset that will be used to train the SCADs model on single component adsorption isotherms data. Experiments will be processed through the following steps:
- **Aggregation of single measurements**
- **Addition of chemical properties based on adsorbent and adsorbate species**
- **Encoding SMILE sequences with a regex-based tokenizer**
- **Conversion of units (Pa for pressure, mol/g for uptake)**
- **Filtering experiments with too few points, out of boundaries values, trailing zeros**
- **Train and validation dataset splitting**

**Model tab:** training, evaluation, and encoding.

through this tab one can train the SCADS model from scratch or resume training for previously trained checkpoints. Moreover, this section provides both model inference and evaluation functionalities. Use the pretrained checkpoint to predict uptake from given experimental condition and pressure. The SCADS model can be evaluated using different metrics, such as:

- **Average mean squared error loss and R square** 
- **Comparison of predicted vs true adsorption isotherms** 

**Viewer tab:** visualization hub.

### 5.1 Setup and Maintenance
`setup_and_maintenance.bat` launches a lightweight maintenance console with these options:

- **Update project**: performs a `git pull` (or fetches release artifacts) so the local checkout stays in sync.
- **Remove logs**: clears `resources/logs` to save disk space or to reset diagnostics before a new run.
- **Open tools**: quick shortcuts to DB Browser for SQLite or other external utilities defined in the script.

### 5.2 Resources
The `NISTADS/resources` tree keeps all mutable assets, making backups and migrations straightforward:

- **checkpoints** — versioned folders containing saved models, training history, evaluation reports, reconstructed samples, and the JSON configuration that produced them. These folders are what you load when resuming training or running inference.
- **configurations** — reusable JSON presets saved through the UI dialogs.
- **database** — includes sub-folders for `images` (training data), `inference` (predicted curves), `metadata` (SQLite records), and `validation` (plots + stats reports).
- **logs** — rotating application logs for troubleshooting. Attach these when reporting issues.
- **templates** — contains `.env` and other templates that need to be copied into write-protected directories (`NISTADS/app`).

Environmental variables reside in `NISTADS/setup/.env`. Copy the template from `resources/templates/.env` and adjust as needed:

| Variable              | Description                                                               |
|-----------------------|---------------------------------------------------------------------------|
| KERAS_BACKEND         | Backend for Keras 3; keep `torch` unless you explicitly need TensorFlow.  |
| TF_CPP_MIN_LOG_LEVEL  | Controls TensorFlow logging verbosity (set to `2` to suppress INFO logs). |
| MPLBACKEND            | Matplotlib backend; `Agg` keeps plotting headless for worker threads.     |

## 6. License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.




