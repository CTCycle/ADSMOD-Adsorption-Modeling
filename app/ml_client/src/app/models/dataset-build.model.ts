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
