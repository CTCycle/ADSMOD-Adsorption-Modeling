export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export interface JsonObject {
    [key: string]: JsonValue;
}

export type InfoModalValue = JsonValue | undefined;
export type InfoModalData = Record<string, InfoModalValue>;

export interface JobStatusResponse {
    job_id: string;
    job_type: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    result?: Record<string, unknown>;
    error?: string;
    poll_interval?: number;
}

export interface JobStartResponse {
    job_id: string;
    job_type: string;
    status: string;
    message: string;
    poll_interval?: number;
}

export type TorchCompileBackend = 'inductor' | 'cudagraphs' | 'aot_eager' | 'eager';

export interface TrainingConfig {
    sample_size?: number;
    validation_size?: number;
    batch_size: number;
    shuffle_dataset: boolean;
    max_buffer_size: number;
    dataset_label?: string;
    dataset_hash?: string | null;
    selected_model: 'SCADS Series' | 'SCADS Atomic';
    dropout_rate: number;
    num_attention_heads: number;
    num_encoders: number;
    molecular_embedding_size: number;
    epochs: number;
    dataloader_workers: number;
    prefetch_factor: number;
    pin_memory: boolean;
    use_device_GPU: boolean;
    device_ID: number;
    use_mixed_precision: boolean;
    use_jit: boolean;
    jit_backend: TorchCompileBackend;
    use_lr_scheduler: boolean;
    initial_lr: number;
    target_lr: number;
    constant_steps: number;
    decay_steps: number;
    save_checkpoints: boolean;
    checkpoints_frequency: number;
    custom_name?: string;
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
    epochs_trained?: number | null;
    final_loss?: number | null;
    final_accuracy?: number | null;
    is_compatible?: boolean;
    created_at?: string;
    configuration: TrainingConfig | null;
    metadata: DatasetFullInfo | null;
    history: JsonObject | null;
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

export type TrainingMetricKey =
    | 'loss'
    | 'val_loss'
    | 'accuracy'
    | 'val_accuracy'
    | 'masked_r2'
    | 'val_masked_r2';

export type TrainingMetrics = Partial<Record<TrainingMetricKey, number>>;

export type TrainingHistoryPoint = {
    epoch: number;
} & TrainingMetrics;

export interface TrainingStatus {
    is_training: boolean;
    current_epoch: number;
    total_epochs: number;
    progress: number;
    metrics?: TrainingMetrics;
    history?: TrainingHistoryPoint[];
    log?: string[];
    poll_interval?: number;
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
    normalization_stats?: JsonObject;
}

export interface ProcessedDatasetInfo {
    dataset_label: string;
    dataset_hash: string | null;
    train_samples: number;
    validation_samples: number;
    created_at?: string;
}
