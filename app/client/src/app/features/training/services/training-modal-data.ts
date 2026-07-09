import type { InfoModalData } from '../../../models/json.model';
import type { CheckpointFullDetails, DatasetFullInfo } from '../../../models/training.model';

export function buildDatasetMetadataModalData(info: DatasetFullInfo): InfoModalData {
    return {
        'Dataset Label': info.dataset_label,
        'Created At': info.created_at,
        'Total Samples': info.total_samples,
        'Train Samples': info.train_samples,
        'Validation Samples': info.validation_samples,
        'Sample Fraction': info.sample_size,
        'Validation Fraction': info.validation_size,
        'Min Measurements': info.min_measurements,
        'Max Measurements': info.max_measurements,
        'SMILES Length': info.smile_sequence_size,
        'Max Pressure': info.max_pressure,
        'Max Uptake': info.max_uptake,
        'SMILES Vocabulary': info.smile_vocabulary_size,
        'Adsorbents Count': info.adsorbent_vocabulary_size,
        Normalization: info.normalization_stats,
    };
}

export function buildCheckpointDetailsModalData(details: CheckpointFullDetails): InfoModalData {
    return {
        Name: details.name,
        'Epochs Trained': details.epochs_trained,
        'Final Loss': details.final_loss?.toFixed(6) ?? 'N/A',
        'Is Compatible': details.is_compatible ? 'Yes' : 'No',
        'Created At': details.created_at || 'Unknown',
    };
}
