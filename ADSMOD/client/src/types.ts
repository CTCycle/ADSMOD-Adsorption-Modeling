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

// Background Job types
export interface JobStartResponse {
    job_id: string;
    job_type: string;
    status: string;
    message: string;
    poll_interval?: number;
}

export interface JobStatusResponse {
    job_id: string;
    job_type: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    result?: Record<string, unknown>;
    error?: string;
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
    sample_size?: number;        // 0.0-1.0 fraction (deprecated)
    validation_size?: number;    // 0.0-1.0 fraction (deprecated)
    batch_size: number;
    shuffle_dataset: boolean;
    shuffle_size: number;
    dataset_label?: string;
    dataset_hash?: string | null;

    // Model settings
    selected_model: 'SCADS Series' | 'SCADS Atomic';
    dropout_rate: number;
    num_attention_heads: number;
    num_encoders: number;
    molecular_embedding_size: number;

    // Training settings
    epochs: number;
    use_device_GPU: boolean;
    use_mixed_precision?: boolean;

    // LR scheduler settings
    use_lr_scheduler: boolean;
    initial_lr: number;
    target_lr: number;
    constant_steps: number;
    decay_steps: number;

    // Callbacks
    save_checkpoints: boolean;
    checkpoints_frequency: number;
    custom_name?: string;
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
    epochs_trained: number | null;
    final_loss: number | null;
    final_accuracy: number | null;
    is_compatible: boolean;
}

export interface CheckpointFullDetails {
    name: string;
    configuration: TrainingConfig | null;
    metadata: DatasetFullInfo | null;
    history: Record<string, unknown> | null;
}

export interface ResumeTrainingConfig {
    checkpoint_name: string;
    additional_epochs: number;
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
    metrics?: Record<string, number>;
    history?: TrainingHistoryPoint[];
    log?: string[];
    poll_interval?: number;
}

export interface TrainingHistoryPoint {
    epoch: number;
    [metric: string]: number | undefined;
}

export interface DatasetSourceInfo {
    source: 'nist' | 'uploaded';
    dataset_name: string;
    display_name: string;
    row_count: number;
}

export interface DatasetSelection {
    source: 'nist' | 'uploaded';
    dataset_name: string;
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
    datasets: DatasetSelection[];
    dataset_label?: string;
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
    dataset_label?: string;
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
    smile_vocabulary_size?: number;
    adsorbent_vocabulary_size?: number;
    normalization_stats?: Record<string, any>;
}

export interface ProcessedDatasetInfo {
    dataset_label: string;
    dataset_hash: string | null;
    train_samples: number;
    validation_samples: number;
    created_at?: string;
}
