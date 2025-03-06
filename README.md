# NISTADS: NIST/ARPA-E dataset composer and modeling

## 1. Project Overview
The NISTADS project delves into the fascinating world of adsorption, a surface-based process where a film of particles (adsorbate) accumulate on the surface of a material (adsorbent). This phenomenon plays a pivotal role in numerous industries. For instance, it is widely applied in water treatment facilities for the purification of water, in air filters to improve air quality, and in the automotive industry within catalytic converters to reduce harmful emissions. The adsorption of compounds is usually quantified by measuring the adsorption isotherm a given adsorbate/adsorbent combination. The objective of this project is two-fold: 1) to collect adsorption isotherms data from the NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials (https://adsorption.nist.gov/index.php#home) through their dedicated API; 2) build a machine learning model that can accurately predict the adsorbed amount of a specific guest-host combination under various conditions, by leveraging the data from the NIST/ARPA-E Database.

This could have significant implications for industries that rely on these materials, potentially leading to more efficient processes and better materials design. As such, this project takes a different approach compared to fitting adsorption data with theoretical model for adsorption constants calculation, instead proposing the use of a deep learning approach to understand adsorption isotherm patterns by leveraging a large volume of experimental data.

![Adsorbent material](NISTADS/commons/assets/5A_with_gas.png)  

## 2. Adsorption datasets
Users can collect data on adsorbent materials and adsorbate species, along with adsorption isotherm experiments. The data is retrieved asynchronously to enhance processing speed. Conveniently, the app will split adsorption isotherm data in two different datasets (Single Component ADSorption and Binary Mixture ADSorption). Since NISTADS is focused on predicting single component adsorption isotherms, it will make use of the single component dataset (SCADS) for the model training. 

As the NIST-ARPA-E database does not provide chemical information for either adsorbate species or adsorbent materials, these details are gathered from external sources. For adsorbate species, NISTADS utilizes the PUG REST API (see PubChemPy documentation for more details) to enrich the dataset with molecular properties such as molecular weight and canonical SMILES. However, obtaining information on adsorbent materials is more challenging, as no publicly available API offers this data. 

The collected data is saved locally in 4 different .csv files. Adsorption isotherm datasets for both ingle component and binary mixture measurements, as well as guest/host properties are saved in *resources/datasets*.

### 2.1 Data preprocessing
The single component adsorption dataset is processed using a tailored pipeline. At first experiments featuring negative values for temperature, pressure, and uptake are removed, together with single measurements falling outside predefined boundaries for pressure and uptake ranges (which can be selected by the user through the configuration file). Pressure and uptakes are standardized to a uniform unit — Pascal for pressure and mol/g for uptakes. Following this refinement, the experiments data is enriched with molecular properties such as molecular weight and SMILE encoding for the adsorbate species and adsorbent materials. Pressure and uptake series are normalized, utilizing upper boundaries as the normalization ceiling, and all sequences are eventually reshaped to have the same length using post-padding with a specified padding value (defaulting to -1 to avoid conflicts with actual values) and then normalized.

## 3. Installation
The installation process on Windows has been designed to be fully automated. To begin, simply run *start_on_windows.bat.* On its first execution, the installation procedure will execute with minimal user input required. The script will check if either Anaconda or Miniconda is installed and can be accessed from your system path. If neither is found, it will automatically download and install the latest Miniconda release from https://docs.anaconda.com/miniconda/. Following this step, the script will proceed with the installation of all necessary Python dependencies. This includes Keras 3 (with PyTorch support as the backend) and the required CUDA dependencies (CUDA 12.4) to enable GPU acceleration. Should you prefer to handle the installation process separately, you can run the standalone installer by running *setup/install_on_windows.bat*.

**Important:** After installation, if the project folder is moved or its path is changed, the application will no longer function correctly. To fix this, you can either:

- Open the main menu, select *Setup and maintentance* and choose *Install project in editable mode*
- Manually run the following commands in the terminal, ensuring the project folder is set as the current working directory (CWD):

    `conda activate NISTADS`

    `pip install -e . --use-pep517` 

### 3.1 Additional Package for XLA Acceleration
This project leverages Just-In-Time model compilation through `torch.compile`, enhancing model performance by tracing the computation graph and applying advanced optimizations like kernel fusion and graph lowering. This approach significantly reduces computation time during both training and inference. The default backend, TorchInductor, is designed to maximize performance on both CPUs and GPUs. Additionally, the installation includes Triton, which generates highly optimized GPU kernels for even faster computation on NVIDIA hardware. For Windows users, a precompiled Triton wheel is bundled with the installation, ensuring seamless integration and performance improvements.

## 4. How to use
On Windows, run *start_on_windows.bat* to launch the main navigation menu and browse through the various options. Please note that some antivirus software, such as Avast, may flag or quarantine python.exe when called by the .bat file. If you encounter unusual behavior, consider adding an exception for your Anaconda or Miniconda environments in your antivirus settings.

**Environmental variables** are stored in *setup/variables/.env*. For security reasons, this file is typically not uploaded to GitHub. Instead, you must create this file manually by copying the template from *resources/templates/.env* and placing it in the *setup/variables* directory.

**KERAS_BACKEND** – Sets the backend for Keras, default is PyTorch.

**TF_CPP_MIN_LOG_LEVEL** – Controls TensorFlow logging verbosity. Setting it to 1 reduces log messages, showing only warnings and errors.

### 4.1 Navigation menu

**1) Data analysis:** perform data validation using a series of metrics for the analysis of the dataset. 

**2) Collect adsorption data:** extract data from the NIST DB and organise them into a readable .csv format. Data is collected through NIST/ARPA-E Database API in a concurrent fashion, allowing for fast data retrieval by selecting a maximum number of parallel HTTP requests. Once the data has been collected, Pug REST API is used to fetch adsorbates molecular properties that will be added to the main dataset. 

**3) Data preprocessing:** prepare the adsorption dataset for machine learning, including normalization of numerical variables and SMILE sequence encoding using a regex-based tokenizer.

**4) Model training and evaluation:** open the machine learning menu to explore various options for model training and validation. Once the menu is open, you will see different options:
- **train from scratch:** start training an instance of the NISTADS model from scratch using the available data and parameters. 
- **train from checkpoint:** start training a pretrained NISTADS checkpoint for an additional amount of epochs, using pretrained model settings and data.  
- **model evaluation:** evaluate the performance of pretrained model checkpoints using different metrics. 

**5) Predict adsorption of compounds:** use the pretrained NISTADS model and predict adsorption of compounds based on pressure.  

**6) Setup and Maintenance:** execute optional commands such as *Install project into environment* to reinstall the project within your environment, *update project* to pull the last updates from github, and *remove logs* to remove all logs saved in *resources/logs*. 

**7) Exit:** close the program immediately 

### 4.2 Resources
This folder is used to organize data and results for various stages of the project, including data validation, model training, and evaluation. Here are the key subfolders:

**datasets:** ..

**predictions:** ..

**checkpoints:** pretrained model checkpoints are stored here, and can be used either for resuming training or performing inference with an already trained model.

## 5. Configurations
For customization, you can modify the main configuration parameters using *settings/configurations.json* 

#### General Configuration
The script is able to perform parallel data fetching through asynchronous HTML requests. However, too many calls may lead to server busy errors, especially when collecting adsorption isotherm data. Try to keep the number of parallel calls for the experiments data below 50 concurrent calls and you should not see any error!

#### General Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| SEED               | Global seed for all numerical operations                 |

#### Data Collection Configuration

| Setting               | Description                                           |
|-------------------------------------------------------------------------------|
| GUEST_FRACTION        | fraction of adsorbate species data to fetch           |
| HOST_FRACTION         | fraction of adsorbent materials data to fetch         |
| EXP_FRACTION          | fraction of adsorption isotherm data to fetch         |
| PARALLEL_TASKS        | parallel calls to get guest/host data                 |
                                 
#### Dataset Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| SAMPLE_SIZE        | Number of samples to use from the dataset                |
| VALIDATION_SIZE    | Proportion of the dataset to use for validation          |
| MAX_PQ_POINTS      | Max number of pressure/uptake points for each experiment |
| MIN_PQ_POINTS      | Min number of pressure/uptake points for each experiment |
| SMILE_PADDING      | Max length of the SMILE sequence                         |
| MAX_PRESSURE       | Max allowed pressure in Pascal                           |
| MAX_UPTAKE         | Max allowed uptake in mol/g                              |
| SPLIT_SEED         | Seed for random splitting of the dataset                 |

#### Model Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| MOLECULAR_EMBEDDING| Embedding dimensions for the molecular properties        |  
| JIT_COMPILE        | Apply Just-In_time (JIT) compiler for model optimization |
| JIT_BACKEND        | Just-In_time (JIT) backend                               |

#### Device Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| DEVICE             | Device to use for training (e.g., GPU)                   |
| DEVICE ID          | ID of the device (only used if GPU is selected)          |
| MIXED_PRECISION    | Whether to use mixed precision training                  |
| NUM_PROCESSORS     | Number of processors to use for data loading             |

#### Training Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| EPOCHS             | Number of epochs to train the model                      |
| ADDITIONAL EPOCHS  | Number of epochs to train the model from checkpoint      |
| LEARNING_RATE      | Learning rate for the optimizer                          |
| BATCH_SIZE         | Number of samples per batch                              |
| USE_TENSORBOARD    | Whether to use TensorBoard for logging                   |
| SAVE_CHECKPOINTS   | Save checkpoints during training (at each epoch)         |

#### LR Scheduler Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| POST_WARMUP_LR     | Learning rate value after initial warmup                 |
| WARMUP_STEPS       | Number of warmup epochs                                  |
            
                 
 
## 6. License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.




