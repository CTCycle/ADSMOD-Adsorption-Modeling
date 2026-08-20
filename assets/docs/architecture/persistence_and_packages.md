# ADSMOD Persistence And Packages

Last updated: 2026-08-20

## Package and persistence ownership

The active backend workspace is `app/server` with one lockfile at
`app/server/uv.lock` and three packages: `shared`, `core_service`, and
`ml_service`.

- `shared.repositories.database.manager.DatabaseManager` owns engine/session
  construction, SQLite pragmas, disposal, and transaction context.
- `shared.repositories.schemas.models.Base.metadata` is the source of truth for
  the twelve operational SQLAlchemy tables.
- Typed repositories in `shared.repositories` own SQL projections and conflict
  targets. `repositories/nist.py` is the sole canonical NIST persistence owner;
  `repositories/queries` contains training-specific query helpers.
- `core_service.services.data.nist_mapper.NISTCanonicalMapper` converts
  provider frames before they enter the shared NIST repository.
- `shared.services.units.UnitRegistry` owns canonical pressure, uptake, and
  temperature parsing/conversion. ML retains DataFrame column orchestration
  only.
- `ml_service` owns model serialization, training execution, and checkpoint
  files, but its active data reads still cross the shared repository boundary.
- `adsmod-core` has a separate immutable `training_snapshots` SQLite store. It
  is not one of the twelve transitional ORM tables.

The active database is managed by the Alembic environment under
`app/server/migrations`. Its reviewed baseline represents the existing
SQLite/PostgreSQL model, and every startup or explicit initialization upgrades
the database to the packaged Alembic head. `Base.metadata` remains the
desired-current-schema authority for autogeneration and drift checks; migration
scripts remain the immutable schema-evolution history.

## Transitional ORM class map

```mermaid
classDiagram
    class CoreServiceContainer {
      +DatabaseManager database
      +DatasetRepository datasets
      +MaterialRepository materials
      +FittingRepository fitting
      +NISTRepository nist_repository
      +DatasetService dataset_service
      +NISTDataService nist_service
      +FittingService fitting_service
    }
    class MlServiceContainer {
      +TrainingManager training_manager
      +TrainingSession training_session
      +TrainingJobRunner training_job_runner
      +TrainingService training_service
    }
    class DatabaseManager
    class BaseMetadata {
      +Base.metadata
    }
    class DatasetRepository
    class MaterialRepository
    class FittingRepository
    class NISTRepository
    class JobManager
    class UnitRegistry {
      +convert_pressure(value, unit, basis)
      +convert_uptake(value, unit, molar_mass)
      +convert_temperature(value, unit)
    }
    class DatasetService
    class NISTDataService
    class FittingService
    class TrainingService
    class TrainingManager
    class TrainingSession
    class TrainingJobRunner
    class CoreContracts
    class MLContracts
    class SharedJobContracts
    class FittingPipeline
    class ModelSolver
    class AdsorptionModels
    class ModelSpec
    class ParameterSpec
    class MetricResult

    CoreServiceContainer --> DatabaseManager
    CoreServiceContainer --> DatasetRepository
    CoreServiceContainer --> MaterialRepository
    CoreServiceContainer --> FittingRepository
    CoreServiceContainer --> NISTRepository
    CoreServiceContainer --> DatasetService
    CoreServiceContainer --> NISTDataService
    CoreServiceContainer --> FittingService
    CoreServiceContainer --> JobManager
    MlServiceContainer --> TrainingManager
    MlServiceContainer --> TrainingSession
    MlServiceContainer --> TrainingJobRunner
    MlServiceContainer --> TrainingService
    MlServiceContainer --> JobManager
    DatasetService --> DatasetRepository
    NISTDataService --> NISTRepository
    NISTDataService --> UnitRegistry
    FittingService --> FittingRepository
    FittingService --> FittingPipeline
    FittingPipeline --> ModelSolver
    FittingPipeline --> AdsorptionModels
    AdsorptionModels --> ModelSpec
    ModelSpec --> ParameterSpec
    ModelSolver --> MetricResult
    TrainingService --> TrainingManager
    TrainingService --> TrainingSession
    TrainingJobRunner --> TrainingSession
    CoreContracts ..> DatasetService
    MLContracts ..> TrainingService
    SharedJobContracts ..> JobManager
    DatabaseManager --> BaseMetadata
```

The `contracts` nodes are Pydantic request/response/workflow types, not ORM
entities. The SQLAlchemy classes below are persistence models and should not be
reused as transport contracts.

## Transitional database ER diagram

The diagram records the current metadata, including nullable foreign keys,
database cascades, relationship cascades, constraints, and the indexes that
matter to repository access. Relationship labels use `CASCADE`, `RESTRICT`, or
`SET NULL` where the model declares an `ondelete` action.

```mermaid
erDiagram
    DATASETS ||--o{ DATASET_IMPORTS : "dataset_id CASCADE"
    DATASETS ||--o{ ISOTHERMS : "dataset_id CASCADE"
    ADSORBENTS ||--o{ ISOTHERMS : "adsorbent_id RESTRICT"
    ISOTHERMS ||--o{ ISOTHERM_COMPONENTS : "isotherm_id CASCADE"
    ADSORBATES ||--o{ ISOTHERM_COMPONENTS : "adsorbate_id RESTRICT"
    ISOTHERMS ||--o{ OBSERVATIONS : "isotherm_id CASCADE"
    ISOTHERM_COMPONENTS ||--o{ OBSERVATIONS : "component_id CASCADE"
    DATASETS ||--o{ FITTING_RUNS : "dataset_id CASCADE"
    ISOTHERMS ||--o{ FITTING_RUNS : "isotherm_id CASCADE"
    ISOTHERM_COMPONENTS ||--o{ FITTING_RUNS : "component_id CASCADE"
    FITTING_RUNS ||--o{ FIT_RESULTS : "run_id CASCADE"
    FIT_RESULTS ||--o{ FIT_PARAMETERS : "result_id CASCADE"
    TRAINING_DATASETS ||--o{ TRAINING_SAMPLES : "training_dataset_id CASCADE"
    ISOTHERMS o|--o{ TRAINING_SAMPLES : "source_isotherm_id SET NULL"

    DATASETS {
        int id PK
        string name "NOT NULL"
        string normalized_name UK "NOT NULL; index source_created"
        string source "uploaded|nist"
        json tags "NOT NULL"
        json provenance "NOT NULL"
        datetime created_at "NOT NULL"
        datetime updated_at "NOT NULL"
        string description "NOT NULL"
        string uq_normalized_name "UNIQUE"
    }
    DATASET_IMPORTS {
        int id PK
        int dataset_id FK "NOT NULL"
        string original_filename "NOT NULL"
        string source_sha256 "NOT NULL"
        string mapping_sha256 "NOT NULL"
        string source_structure "atomic|aggregated|mixed"
        string parser_version "NOT NULL"
        json column_mapping "NOT NULL"
        json validation_result "NOT NULL"
        json warnings "NOT NULL"
        string uq_dataset_source "UNIQUE"
    }
    ADSORBATES {
        int id PK
        string key UK "NOT NULL"
        string name "NOT NULL"
        string normalized_name "NOT NULL; index"
        string inchi_key UK "NULLABLE"
        string inchi "NULLABLE"
        string formula "NULLABLE"
        float molar_mass_g_mol "NULLABLE"
        string smiles "NULLABLE"
    }
    ADSORBENTS {
        int id PK
        string key UK "NOT NULL"
        string name "NOT NULL"
        string normalized_name "NOT NULL; index"
        string external_identifier UK "NULLABLE"
        string formula "NULLABLE"
        float molar_mass_g_mol "NULLABLE"
        string smiles "NULLABLE"
    }
    ISOTHERMS {
        int id PK
        int dataset_id FK "NOT NULL"
        string external_key "NOT NULL"
        string name "NOT NULL"
        int adsorbent_id FK "NOT NULL"
        float temperature_k "NOT NULL; > 0"
        string temperature_original_unit "NOT NULL"
        string pressure_basis "absolute|partial|relative"
        string duplicate_policy "reject|keep|average"
        float saturation_pressure_pa "NULLABLE; > 0 when set"
        string uq_dataset_external_key "UNIQUE"
        string ix_dataset_name "INDEX"
    }
    ISOTHERM_COMPONENTS {
        int id PK
        int isotherm_id FK "NOT NULL"
        int position "NOT NULL; >= 1"
        int adsorbate_id FK "NOT NULL"
        float mole_fraction "NULLABLE; 0..1"
        string uq_isotherm_position "UNIQUE"
        string uq_isotherm_adsorbate "UNIQUE"
    }
    OBSERVATIONS {
        int id PK
        int isotherm_id FK "NOT NULL"
        int component_id FK "NOT NULL"
        int sequence_index "NOT NULL; >= 0"
        int source_row "NULLABLE"
        float pressure_canonical "NOT NULL; >= 0"
        string pressure_canonical_unit "Pa|1"
        float uptake_mol_kg "NOT NULL; >= 0"
        float uptake_stddev_mol_kg "NULLABLE; > 0"
        json conversion_metadata "NOT NULL"
        json extra_metadata "NOT NULL"
        string uq_isotherm_component_sequence "UNIQUE"
        string ix_component_pressure "INDEX"
    }
    FITTING_RUNS {
        int id PK
        int dataset_id FK "NOT NULL"
        int isotherm_id FK "NOT NULL"
        int component_id FK "NOT NULL"
        string input_sha256 "NOT NULL"
        string optimizer "NOT NULL"
        int max_evaluations "NOT NULL; > 0"
        string pressure_display_unit "NOT NULL"
        string uptake_display_unit "NOT NULL"
        json configuration "NOT NULL"
        string status "running|completed|warning|failed"
        string message "NOT NULL"
        int best_result_id "NULLABLE; no FK currently"
        string ix_isotherm_created "INDEX"
    }
    FIT_RESULTS {
        int id PK
        int run_id FK "NOT NULL"
        string model_name "NOT NULL"
        string model_version "NOT NULL"
        string status "success|warning|failed"
        int observation_count "NOT NULL; >= 0"
        int parameter_count "NOT NULL; > 0"
        float sse "NULLABLE"
        float rmse "NULLABLE"
        float mae "NULLABLE"
        json predicted_observations "NOT NULL"
        json predicted_curve "NOT NULL"
        int rank "NULLABLE"
        string uq_run_model "UNIQUE"
        string ix_run_rank "INDEX"
    }
    FIT_PARAMETERS {
        int result_id PK,FK
        string name PK
        int position "NOT NULL; >= 0"
        float value_canonical "NOT NULL"
        float standard_error_canonical "NULLABLE"
        string unit_canonical "NOT NULL"
        string uq_result_position "UNIQUE"
    }
    TRAINING_DATASETS {
        int id PK
        string content_hash UK "NOT NULL"
        string label "NOT NULL"
        json configuration "NOT NULL"
        float sample_fraction "NOT NULL; 0..1"
        float validation_fraction "NOT NULL; 0..1"
        int min_measurements "NULLABLE"
        int max_measurements "NULLABLE"
        int total_samples "NOT NULL"
        int train_samples "NOT NULL"
        int validation_samples "NOT NULL"
        int test_samples "NOT NULL"
        json vocabularies "NOT NULL"
        json normalization_stats "NOT NULL"
    }
    TRAINING_SAMPLES {
        int id PK
        int training_dataset_id FK "NOT NULL"
        string sample_key "NOT NULL"
        int source_isotherm_id FK "NULLABLE; SET NULL"
        string split "train|validation|test"
        float temperature_k "NOT NULL"
        json pressure_values "NOT NULL"
        json uptake_values "NOT NULL"
        int encoded_adsorbent "NULLABLE"
        float adsorbate_molar_mass "NULLABLE"
        json encoded_smiles "NULLABLE"
        string uq_dataset_sample_key "UNIQUE"
        string ix_dataset_split "INDEX"
    }
```

## v3 snapshot store

The extracted core package owns a separate SQLite file for immutable training
snapshots. The ML package accesses it only through authenticated HTTP, not by
opening the file or importing the core package.

```mermaid
flowchart LR
    Core["adsmod-core snapshot API"] --> Store[("training_snapshots.sqlite")]
    Store --> Snapshot["snapshot_id + SHA-256 + paginated rows"]
    ML["adsmod-ml CoreSnapshotClient"] -. "token + hash verification" .-> Core
    ML --> Training["future ML training input"]
```

This store is the target replacement for ML reads from the transitional shared
database. Snapshot retention is intentionally deferred until restart durability
or retention requirements are demonstrated.

## Validation expectations

- Run schema-contract tests against `Base.metadata` and the generated OpenAPI
  snapshots from the running apps.
- Regenerate `app/resources/adsmod.schema.json` from `AdsmodConfig`; a second
  generation must produce no diff.
- Do not hand-edit either generated schema or OpenAPI snapshots.
- Before schema changes, add the Alembic baseline described above. A later
  migration should make `FittingRun.best_result_id` a real foreign key and
  either remove or constrain redundant dataset/isotherm/component references.
