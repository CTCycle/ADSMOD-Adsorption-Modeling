// Type definitions for ADSMOD frontend

export interface DatasetPayload {
    dataset_name: string;
    columns: string[];
    records: Record<string, unknown>[];
}

export interface ParameterBound {
    min: number;
    max: number;
}

export interface ModelParameters {
    [parameterName: string]: ParameterBound;
}

export interface ModelConfiguration {
    min: Record<string, number>;
    max: Record<string, number>;
    initial: Record<string, number>;
}

export interface FittingPayload {
    max_iterations: number;
    optimization_method: 'LSS' | 'BFGS' | 'L-BFGS-B' | 'Nelder-Mead' | 'Powell';
    parameter_bounds: Record<string, ModelConfiguration>;
    dataset: DatasetPayload;
}

export interface DatasetResponse {
    status: string;
    dataset?: DatasetPayload;
    summary?: string;
    detail?: string;
    message?: string;
}

export interface FittingResponse {
    status: string;
    summary?: string;
    detail?: string;
    message?: string;
    processed_rows?: number;
    best_model_saved?: boolean;
    models?: string[];
}

// NIST API types
export interface NISTFetchRequest {
    experiments_fraction: number;
    guest_fraction: number;
    host_fraction: number;
}

export interface NISTFetchResponse {
    status: string;
    experiments_count: number;
    single_component_rows: number;
    binary_mixture_rows: number;
    guest_rows: number;
    host_rows: number;
    detail?: string;
    message?: string;
}

export interface NISTPropertiesRequest {
    target: 'guest' | 'host';
}

export interface NISTPropertiesResponse {
    status: string;
    target: string;
    names_requested: number;
    names_matched: number;
    rows_updated: number;
    detail?: string;
    message?: string;
}

export interface NISTStatusResponse {
    status: string;
    data_available: boolean;
    single_component_rows: number;
    binary_mixture_rows: number;
    guest_rows: number;
    host_rows: number;
    detail?: string;
    message?: string;
}

export type ParameterKey = [string, string, string]; // [model, parameter, bound_type]

// Browser API types
export interface TableInfo {
    table_name: string;
    display_name: string;
}

export interface TableListResponse {
    status: string;
    tables: TableInfo[];
}

export interface TableDataResponse {
    status: string;
    table_name: string;
    display_name: string;
    row_count: number;
    column_count: number;
    columns: string[];
    data: Record<string, unknown>[];
}

// Training API types
export interface TrainingConfig {
    // Dataset settings
    sample_size: number;        // 0.0-1.0 fraction
    validation_size: number;    // 0.0-1.0 fraction
    batch_size: number;
    shuffle_dataset: boolean;
    shuffle_size: number;

    // Model settings
    selected_model: 'SCADS Series' | 'SCADS Atomic';
    dropout_rate: number;
    num_attention_heads: number;
    num_encoders: number;
    molecular_embedding_size: number;

    // Training settings
    epochs: number;

    // LR scheduler settings
    use_lr_scheduler: boolean;
    initial_lr: number;
    target_lr: number;
    constant_steps: number;
    decay_steps: number;

    // Callbacks
    save_checkpoints: boolean;
    checkpoints_frequency: number;
}

export interface TrainingUpdate {
    type: 'log' | 'metric' | 'progress' | 'status';
    data: {
        message?: string;
        epoch?: number;
        batch?: number;
        loss?: number;
        accuracy?: number;
        val_loss?: number;
        val_accuracy?: number;
        progress?: number;
        status?: 'training' | 'completed' | 'stopped' | 'error';
    };
}

export interface CheckpointInfo {
    name: string;
    created_at?: string;
    epochs_trained?: number;
    final_loss?: number;
    final_accuracy?: number;
}

export interface TrainingDatasetInfo {
    available: boolean;
    name?: string;
    train_samples?: number;
    validation_samples?: number;
}

export interface TrainingStatus {
    is_training: boolean;
    current_epoch: number;
    total_epochs: number;
    progress: number;
}

// Dataset Builder types
export interface DatasetBuildConfig {
    sample_size: number;
    validation_size: number;
    min_measurements: number;
    max_measurements: number;
    smile_sequence_size: number;
    max_pressure: number;
    max_uptake: number;
    source_datasets: string[];
}

export interface DatasetBuildResult {
    success: boolean;
    message: string;
    total_samples?: number;
    train_samples?: number;
    validation_samples?: number;
}

export interface DatasetFullInfo {
    available: boolean;
    created_at?: string;
    sample_size?: number;
    validation_size?: number;
    min_measurements?: number;
    max_measurements?: number;
    smile_sequence_size?: number;
    max_pressure?: number;
    max_uptake?: number;
    total_samples?: number;
    train_samples?: number;
    validation_samples?: number;
}
