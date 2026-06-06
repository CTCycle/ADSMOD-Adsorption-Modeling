import { API_BASE_URL } from '../core/config/api-base-url';
import type { JsonObject } from '../models/json.model';
import type {
    CheckpointFullDetails,
    CheckpointInfo,
    DatasetFullInfo,
    DatasetSourceInfo,
    ResumeTrainingConfig,
    TrainingConfig,
    TrainingDatasetInfo,
    TrainingHistoryPoint,
    TrainingMetricKey,
    TrainingMetrics,
    TrainingStatus,
} from '../models/training.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';

type UnknownRecord = Record<string, unknown>;

const asRecord = (value: unknown): UnknownRecord | null => (
    value !== null && typeof value === 'object' ? (value as UnknownRecord) : null
);

const getString = (record: UnknownRecord | null, key: string): string | undefined => {
    const value = record?.[key];
    return typeof value === 'string' ? value : undefined;
};

const getNumber = (record: UnknownRecord | null, key: string): number | undefined => {
    const value = record?.[key];
    return typeof value === 'number' ? value : undefined;
};

const getBoolean = (record: UnknownRecord | null, key: string): boolean | undefined => {
    const value = record?.[key];
    return typeof value === 'boolean' ? value : undefined;
};

const parseCheckpointFullDetails = (record: UnknownRecord): CheckpointFullDetails => ({
    name: getString(record, 'name') ?? 'Unknown checkpoint',
    epochs_trained: getNumber(record, 'epochs_trained') ?? null,
    final_loss: getNumber(record, 'final_loss') ?? null,
    final_accuracy: getNumber(record, 'final_accuracy') ?? null,
    is_compatible: getBoolean(record, 'is_compatible'),
    created_at: getString(record, 'created_at'),
    configuration: asRecord(record['configuration']) ? (record['configuration'] as TrainingConfig) : null,
    metadata: asRecord(record['metadata']) ? (record['metadata'] as DatasetFullInfo) : null,
    history: (asRecord(record['history']) as JsonObject | null),
});

async function startTrainingSession(
    endpoint: string,
    config: TrainingConfig | ResumeTrainingConfig,
    defaultMessage: string
): Promise<{ sessionId: string; message: string; status: 'started' | 'error'; poll_interval?: number }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}${endpoint}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            },
            HTTP_TIMEOUT
        );
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { sessionId: '', message: extractErrorMessage(response, data), status: 'error' };
        }

        const result = asRecord(await response.json().catch(() => null));
        return {
            sessionId: getString(result, 'session_id') ?? '',
            message: getString(result, 'message') ?? defaultMessage,
            status: 'started',
            poll_interval: getNumber(result, 'poll_interval'),
        };
    } catch (error) {
        return {
            sessionId: '',
            message: error instanceof Error ? error.message : 'An unknown error occurred.',
            status: 'error',
        };
    }
}

const TRAINING_METRIC_KEYS: readonly TrainingMetricKey[] = [
    'loss',
    'val_loss',
    'accuracy',
    'val_accuracy',
    'masked_r2',
    'val_masked_r2',
];

const toFiniteNumber = (value: unknown): number | undefined => (
    typeof value === 'number' && Number.isFinite(value) ? value : undefined
);

const parseTrainingMetrics = (value: unknown): TrainingMetrics => {
    if (!value || typeof value !== 'object') {
        return {};
    }

    const source = value as Record<string, unknown>;
    const metrics: TrainingMetrics = {};
    for (const metricKey of TRAINING_METRIC_KEYS) {
        const metricValue = toFiniteNumber(source[metricKey]);
        if (metricValue !== undefined) {
            metrics[metricKey] = metricValue;
        }
    }

    return metrics;
};

const parseTrainingHistory = (value: unknown): TrainingHistoryPoint[] => {
    if (!Array.isArray(value)) {
        return [];
    }

    const history: TrainingHistoryPoint[] = [];
    for (const historyEntry of value) {
        if (!historyEntry || typeof historyEntry !== 'object') {
            continue;
        }

        const source = historyEntry as Record<string, unknown>;
        const epoch = toFiniteNumber(source['epoch']);
        if (epoch === undefined) {
            continue;
        }

        history.push({
            epoch,
            ...parseTrainingMetrics(source),
        });
    }

    return history;
};

export async function fetchTrainingDatasets(): Promise<{ data: TrainingDatasetInfo; error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/training/datasets`, { method: 'GET' }, HTTP_TIMEOUT);
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { data: { available: false }, error: extractErrorMessage(response, data) };
        }

        const result = asRecord(await response.json().catch(() => null));
        return {
            data: {
                available: getBoolean(result, 'available') ?? false,
                name: getString(result, 'name'),
                train_samples: getNumber(result, 'train_samples'),
                validation_samples: getNumber(result, 'validation_samples'),
            },
            error: null,
        };
    } catch (error) {
        return { data: { available: false }, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}

export async function fetchDatasetSources(): Promise<{ datasets: DatasetSourceInfo[]; error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/training/dataset-sources`, { method: 'GET' }, HTTP_TIMEOUT);
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { datasets: [], error: extractErrorMessage(response, data) };
        }

        const result = asRecord(await response.json().catch(() => null));
        const rawDatasets = Array.isArray(result?.['datasets']) ? result['datasets'] : [];
        const datasets: DatasetSourceInfo[] = rawDatasets
            .map((dataset): DatasetSourceInfo | null => {
                const datasetRecord = asRecord(dataset);
                const source = getString(datasetRecord, 'source');
                if (source !== 'nist' && source !== 'uploaded') {
                    return null;
                }

                const datasetName = getString(datasetRecord, 'dataset_name');
                const displayName = getString(datasetRecord, 'display_name');
                const rowCount = getNumber(datasetRecord, 'row_count');
                if (!datasetName || !displayName || rowCount === undefined) {
                    return null;
                }

                return { source, dataset_name: datasetName, display_name: displayName, row_count: rowCount };
            })
            .filter((dataset): dataset is DatasetSourceInfo => dataset !== null);

        return { datasets, error: null };
    } catch (error) {
        return { datasets: [], error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}

export async function fetchCheckpoints(): Promise<{ checkpoints: CheckpointInfo[]; error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/training/checkpoints`, { method: 'GET' }, HTTP_TIMEOUT);
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { checkpoints: [], error: extractErrorMessage(response, data) };
        }

        const result = asRecord(await response.json().catch(() => null));
        const rawCheckpoints = Array.isArray(result?.['checkpoints']) ? result['checkpoints'] : [];
        const checkpoints: CheckpointInfo[] = rawCheckpoints.map((checkpoint) => {
            const record = asRecord(checkpoint);
            return {
                name: getString(record, 'name') ?? 'Unknown checkpoint',
                epochs_trained: getNumber(record, 'epochs_trained') ?? null,
                final_loss: getNumber(record, 'final_loss') ?? null,
                final_accuracy: getNumber(record, 'final_accuracy') ?? null,
                is_compatible: getBoolean(record, 'is_compatible') ?? false,
            };
        });

        return { checkpoints, error: null };
    } catch (error) {
        return { checkpoints: [], error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}

export async function fetchCheckpointDetails(
    checkpointName: string
): Promise<{ details: CheckpointFullDetails | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/checkpoints/${encodeURIComponent(checkpointName)}`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );
        if (!response.ok) {
            throw new Error(`Failed to fetch checkpoint details: ${response.statusText}`);
        }

        const data = asRecord(await response.json().catch(() => null));
        if (!data) {
            return { details: null, error: 'Invalid checkpoint details response.' };
        }

        return { details: parseCheckpointFullDetails(data), error: null };
    } catch (error) {
        console.error('Error fetching checkpoint details:', error);
        return { details: null, error: error instanceof Error ? error.message : 'Unknown error' };
    }
}

export async function deleteCheckpoint(
    checkpointName: string
): Promise<{ success: boolean; error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/checkpoints/${encodeURIComponent(checkpointName)}`,
            { method: 'DELETE' },
            HTTP_TIMEOUT
        );
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || `Failed to delete checkpoint: ${response.statusText}`);
        }

        return { success: true, error: null };
    } catch (error) {
        console.error('Error deleting checkpoint:', error);
        return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
}

export async function startTraining(config: TrainingConfig): Promise<{
    sessionId: string;
    message: string;
    status: 'started' | 'error';
    poll_interval?: number;
}> {
    return startTrainingSession('/training/start', config, 'Training started.');
}

export async function resumeTraining(config: ResumeTrainingConfig): Promise<{
    sessionId: string;
    message: string;
    status: 'started' | 'error';
    poll_interval?: number;
}> {
    return startTrainingSession('/training/resume', config, 'Training resumed.');
}

export async function stopTraining(): Promise<{ message: string; status: 'stopped' | 'error' }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/training/stop`, { method: 'POST' }, HTTP_TIMEOUT);
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { message: extractErrorMessage(response, data), status: 'error' };
        }

        const result = asRecord(await response.json().catch(() => null));
        return { message: getString(result, 'message') ?? 'Training stopped.', status: 'stopped' };
    } catch (error) {
        return { message: error instanceof Error ? error.message : 'An unknown error occurred.', status: 'error' };
    }
}

export async function getTrainingStatus(): Promise<TrainingStatus & { error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/training/status`, { method: 'GET' }, HTTP_TIMEOUT);
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return {
                is_training: false,
                current_epoch: 0,
                total_epochs: 0,
                progress: 0,
                metrics: {},
                history: [],
                log: [],
                error: extractErrorMessage(response, data),
            };
        }

        const result = asRecord(await response.json().catch(() => null));
        return {
            is_training: getBoolean(result, 'is_training') ?? false,
            current_epoch: getNumber(result, 'current_epoch') ?? 0,
            total_epochs: getNumber(result, 'total_epochs') ?? 0,
            progress: getNumber(result, 'progress') ?? 0,
            metrics: parseTrainingMetrics(result?.['metrics']),
            history: parseTrainingHistory(result?.['history']),
            log: Array.isArray(result?.['log'])
                ? result['log'].filter((entry): entry is string => typeof entry === 'string')
                : [],
            poll_interval: getNumber(result, 'poll_interval'),
            error: null,
        };
    } catch (error) {
        return {
            is_training: false,
            current_epoch: 0,
            total_epochs: 0,
            progress: 0,
            metrics: {},
            history: [],
            log: [],
            error: error instanceof Error ? error.message : 'An unknown error occurred.',
        };
    }
}
