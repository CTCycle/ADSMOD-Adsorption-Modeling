import type {
    CheckpointInfo,
    ResumeTrainingConfig,
    TrainingConfig,
    TrainingDatasetInfo,
    TrainingStatus,
    DatasetSourceInfo,
} from '../types';
import { API_BASE_URL } from '../constants';
import { fetchWithTimeout, extractErrorMessage, HTTP_TIMEOUT } from './http';

export async function fetchTrainingDatasets(): Promise<{
    data: TrainingDatasetInfo;
    error: string | null;
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/datasets`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { data: { available: false }, error: message };
        }

        const result = await response.json();
        return {
            data: {
                available: result.available || false,
                name: result.name,
                train_samples: result.train_samples,
                validation_samples: result.validation_samples,
            },
            error: null,
        };
    } catch (error) {
        if (error instanceof Error) {
            return { data: { available: false }, error: error.message };
        }
        return { data: { available: false }, error: 'An unknown error occurred.' };
    }
}

export async function fetchDatasetSources(): Promise<{
    datasets: DatasetSourceInfo[];
    error: string | null;
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/dataset-sources`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { datasets: [], error: message };
        }

        const result = await response.json();
        return {
            datasets: result.datasets || [],
            error: null,
        };
    } catch (error) {
        if (error instanceof Error) {
            return { datasets: [], error: error.message };
        }
        return { datasets: [], error: 'An unknown error occurred.' };
    }
}

export async function fetchCheckpoints(): Promise<{
    checkpoints: CheckpointInfo[];
    error: string | null;
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/checkpoints`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { checkpoints: [], error: message };
        }

        const result = await response.json();
        const rawCheckpoints = Array.isArray(result.checkpoints) ? result.checkpoints : [];
        const checkpoints: CheckpointInfo[] = rawCheckpoints.map((checkpoint: unknown) => {
            if (typeof checkpoint === 'string') {
                return {
                    name: checkpoint,
                    epochs_trained: null,
                    final_loss: null,
                    final_accuracy: null,
                    is_compatible: false,
                };
            }
            const record = checkpoint as Partial<CheckpointInfo>;
            return {
                name: record.name || 'Unknown checkpoint',
                epochs_trained: record.epochs_trained ?? null,
                final_loss: record.final_loss ?? null,
                final_accuracy: record.final_accuracy ?? null,
                is_compatible: record.is_compatible ?? false,
            };
        });
        return { checkpoints, error: null };
    } catch (error) {
        if (error instanceof Error) {
            return { checkpoints: [], error: error.message };
        }
        return { checkpoints: [], error: 'An unknown error occurred.' };
    }
}

export async function startTraining(config: TrainingConfig): Promise<{
    sessionId: string;
    message: string;
    status: 'started' | 'error';
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/start`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { sessionId: '', message, status: 'error' };
        }

        const result = await response.json();
        return {
            sessionId: result.session_id || '',
            message: result.message || 'Training started.',
            status: 'started',
        };
    } catch (error) {
        if (error instanceof Error) {
            return { sessionId: '', message: error.message, status: 'error' };
        }
        return { sessionId: '', message: 'An unknown error occurred.', status: 'error' };
    }
}

export async function resumeTraining(config: ResumeTrainingConfig): Promise<{
    sessionId: string;
    message: string;
    status: 'started' | 'error';
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/resume`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { sessionId: '', message, status: 'error' };
        }

        const result = await response.json();
        return {
            sessionId: result.session_id || '',
            message: result.message || 'Training resumed.',
            status: 'started',
        };
    } catch (error) {
        if (error instanceof Error) {
            return { sessionId: '', message: error.message, status: 'error' };
        }
        return { sessionId: '', message: 'An unknown error occurred.', status: 'error' };
    }
}

export async function stopTraining(): Promise<{
    message: string;
    status: 'stopped' | 'error';
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/stop`,
            { method: 'POST' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { message, status: 'error' };
        }

        const result = await response.json();
        return {
            message: result.message || 'Training stopped.',
            status: 'stopped',
        };
    } catch (error) {
        if (error instanceof Error) {
            return { message: error.message, status: 'error' };
        }
        return { message: 'An unknown error occurred.', status: 'error' };
    }
}

export async function getTrainingStatus(): Promise<TrainingStatus & { error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/status`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return {
                is_training: false,
                current_epoch: 0,
                total_epochs: 0,
                progress: 0,
                metrics: {},
                history: [],
                log: [],
                error: message,
            };
        }

        const result = await response.json();
        return {
            is_training: result.is_training || false,
            current_epoch: result.current_epoch || 0,
            total_epochs: result.total_epochs || 0,
            progress: result.progress || 0,
            metrics: typeof result.metrics === 'object' && result.metrics !== null ? result.metrics : {},
            history: Array.isArray(result.history) ? result.history : [],
            log: Array.isArray(result.log) ? result.log : [],
            error: null,
        };
    } catch (error) {
        if (error instanceof Error) {
            return {
                is_training: false,
                current_epoch: 0,
                total_epochs: 0,
                progress: 0,
                metrics: {},
                history: [],
                log: [],
                error: error.message,
            };
        }
        return {
            is_training: false,
            current_epoch: 0,
            total_epochs: 0,
            progress: 0,
            metrics: {},
            history: [],
            log: [],
            error: 'An unknown error occurred.',
        };
    }
}
