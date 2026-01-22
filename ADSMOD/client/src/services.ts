// API service for dataset and fitting endpoints

import type {
    DatasetPayload,
    DatasetResponse,
    FittingPayload,
    FittingResponse,
    JobStartResponse,
    JobStatusResponse,
    NISTFetchRequest,
    NISTPropertiesRequest,
    NISTStatusResponse,
} from './types';
import { API_BASE_URL } from './constants';

const HTTP_TIMEOUT = 120000; // 120 seconds
const NIST_TIMEOUT = 300000; // 5 minutes for larger NIST collections

async function fetchWithTimeout(url: string, options: RequestInit, timeout: number): Promise<Response> {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
        });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        throw error;
    }
}

function extractErrorMessage(response: Response, data: unknown): string {
    if (typeof data === 'object' && data !== null) {
        const obj = data as Record<string, unknown>;
        if (typeof obj.detail === 'string' && obj.detail) {
            return obj.detail;
        }
        if (typeof obj.message === 'string' && obj.message) {
            return obj.message;
        }
    }
    return `HTTP error ${response.status}`;
}

export async function loadDataset(file: File): Promise<{ dataset: DatasetPayload | null; message: string }> {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/datasets/load`,
            {
                method: 'POST',
                body: formData,
            },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { dataset: null, message: `[ERROR] ${message}` };
        }

        const data = (await response.json()) as DatasetResponse;

        if (data.status !== 'success') {
            const detail = data.detail || 'Failed to load dataset.';
            return { dataset: null, message: `[ERROR] ${detail}` };
        }

        if (!data.dataset) {
            return { dataset: null, message: '[ERROR] Backend returned an invalid dataset payload.' };
        }

        const summary = data.summary || '[INFO] Dataset loaded successfully.';
        return { dataset: data.dataset, message: summary };
    } catch (error) {
        if (error instanceof Error) {
            return { dataset: null, message: `[ERROR] Failed to reach ADSMOD backend: ${error.message}` };
        }
        return { dataset: null, message: '[ERROR] An unknown error occurred.' };
    }
}

export async function fetchDatasetNames(): Promise<{ names: string[]; error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/datasets/names`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { names: [], error: message };
        }

        const result = await response.json();
        return { names: result.names || [], error: null };
    } catch (error) {
        if (error instanceof Error) {
            return { names: [], error: error.message };
        }
        return { names: [], error: 'An unknown error occurred.' };
    }
}

export async function fetchDatasetByName(datasetName: string): Promise<{
    dataset: DatasetPayload | null;
    summary: string | null;
    error: string | null;
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/datasets/by-name/${encodeURIComponent(datasetName)}`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { dataset: null, summary: null, error: message };
        }

        const data = (await response.json()) as DatasetResponse;
        if (data.status !== 'success') {
            const detail = data.detail || data.message || 'Failed to load dataset.';
            return { dataset: null, summary: null, error: detail };
        }

        return { dataset: data.dataset || null, summary: data.summary || null, error: null };
    } catch (error) {
        if (error instanceof Error) {
            return { dataset: null, summary: null, error: error.message };
        }
        return { dataset: null, summary: null, error: 'An unknown error occurred.' };
    }
}

export async function fetchNistDataForFitting(): Promise<{ dataset: DatasetPayload | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/fitting/nist-dataset`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { dataset: null, error: message };
        }

        const result = await response.json();
        if (result.status !== 'success') {
            const detail = result.detail || result.message || 'Failed to load NIST data.';
            return { dataset: null, error: detail };
        }

        return { dataset: result.dataset || null, error: null };
    } catch (error) {
        if (error instanceof Error) {
            return { dataset: null, error: error.message };
        }
        return { dataset: null, error: 'An unknown error occurred.' };
    }
}

// Job polling utilities
const JOB_POLL_INTERVAL = 1000; // 1 second

export async function pollJobStatus(
    endpoint: string,
    jobId: string
): Promise<JobStatusResponse | null> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}${endpoint}/jobs/${jobId}`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );
        if (!response.ok) {
            return null;
        }
        return (await response.json()) as JobStatusResponse;
    } catch {
        return null;
    }
}

export async function startFittingJob(
    payload: FittingPayload
): Promise<{ jobId: string | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/fitting/run`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { jobId: null, error: message };
        }

        const result = (await response.json()) as JobStartResponse;
        return { jobId: result.job_id, error: null };
    } catch (error) {
        if (error instanceof Error) {
            return { jobId: null, error: error.message };
        }
        return { jobId: null, error: 'An unknown error occurred.' };
    }
}

export async function pollFittingJobUntilComplete(
    jobId: string,
    onProgress?: (status: JobStatusResponse) => void
): Promise<{ message: string; data: FittingResponse | null }> {
    while (true) {
        const status = await pollJobStatus('/fitting', jobId);
        if (!status) {
            return { message: '[ERROR] Failed to poll job status.', data: null };
        }

        if (onProgress) {
            onProgress(status);
        }

        if (status.status === 'completed') {
            const result = status.result as FittingResponse | undefined;
            if (!result) {
                return { message: '[INFO] Fitting completed.', data: null };
            }
            const lines: string[] = ['[INFO] Fitting completed successfully.'];
            if (typeof result.processed_rows === 'number') {
                lines.push(`Processed experiments: ${result.processed_rows}`);
            }
            return { message: result.summary || lines.join('\n'), data: result };
        }

        if (status.status === 'failed') {
            return { message: `[ERROR] ${status.error || 'Job failed.'}`, data: null };
        }

        if (status.status === 'cancelled') {
            return { message: '[INFO] Job was cancelled.', data: null };
        }

        await new Promise((resolve) => setTimeout(resolve, JOB_POLL_INTERVAL));
    }
}

// Legacy startFitting that uses job polling internally
export async function startFitting(
    payload: FittingPayload,
    onProgress?: (status: JobStatusResponse) => void
): Promise<{ message: string; data: FittingResponse | null }> {
    const { jobId, error } = await startFittingJob(payload);
    if (error || !jobId) {
        return { message: `[ERROR] ${error || 'Failed to start job.'}`, data: null };
    }
    return pollFittingJobUntilComplete(jobId, onProgress);
}

export async function startNistFetchJob(
    payload: NISTFetchRequest
): Promise<{ jobId: string | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/nist/fetch`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { jobId: null, error: message };
        }

        const result = (await response.json()) as JobStartResponse;
        return { jobId: result.job_id, error: null };
    } catch (error) {
        if (error instanceof Error) {
            return { jobId: null, error: error.message };
        }
        return { jobId: null, error: 'An unknown error occurred.' };
    }
}

export async function startNistPropertiesJob(
    payload: NISTPropertiesRequest
): Promise<{ jobId: string | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/nist/properties`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { jobId: null, error: message };
        }

        const result = (await response.json()) as JobStartResponse;
        return { jobId: result.job_id, error: null };
    } catch (error) {
        if (error instanceof Error) {
            return { jobId: null, error: error.message };
        }
        return { jobId: null, error: 'An unknown error occurred.' };
    }
}

export async function pollNistJobUntilComplete(
    jobId: string,
    onProgress?: (status: JobStatusResponse) => void
): Promise<{ result: Record<string, unknown> | null; error: string | null }> {
    while (true) {
        const status = await pollJobStatus('/nist', jobId);
        if (!status) {
            return { result: null, error: 'Failed to poll job status.' };
        }

        if (onProgress) {
            onProgress(status);
        }

        if (status.status === 'completed') {
            return { result: status.result || null, error: null };
        }

        if (status.status === 'failed') {
            return { result: null, error: status.error || 'Job failed.' };
        }

        if (status.status === 'cancelled') {
            return { result: null, error: 'Job was cancelled.' };
        }

        await new Promise((resolve) => setTimeout(resolve, JOB_POLL_INTERVAL));
    }
}

// Legacy fetchNistData that uses job polling internally
export async function fetchNistData(
    payload: NISTFetchRequest,
    onProgress?: (status: JobStatusResponse) => void
): Promise<{ data: Record<string, unknown> | null; error: string | null }> {
    const { jobId, error: startError } = await startNistFetchJob(payload);
    if (startError || !jobId) {
        return { data: null, error: startError || 'Failed to start job.' };
    }
    const { result, error } = await pollNistJobUntilComplete(jobId, onProgress);
    return { data: result, error };
}

// Legacy fetchNistProperties that uses job polling internally
export async function fetchNistProperties(
    payload: NISTPropertiesRequest,
    onProgress?: (status: JobStatusResponse) => void
): Promise<{ data: Record<string, unknown> | null; error: string | null }> {
    const { jobId, error: startError } = await startNistPropertiesJob(payload);
    if (startError || !jobId) {
        return { data: null, error: startError || 'Failed to start job.' };
    }
    const { result, error } = await pollNistJobUntilComplete(jobId, onProgress);
    return { data: result, error };
}

export async function fetchNistStatus(): Promise<{
    data: NISTStatusResponse | null;
    error: string | null;
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/nist/status`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { data: null, error: message };
        }

        const result = (await response.json()) as NISTStatusResponse;
        if (result.status !== 'success') {
            const detail = result.detail || result.message || 'Failed to load NIST status.';
            return { data: result, error: detail };
        }

        return { data: result, error: null };
    } catch (error) {
        if (error instanceof Error) {
            return { data: null, error: error.message };
        }
        return { data: null, error: 'An unknown error occurred.' };
    }
}

export async function fetchTableList(): Promise<{ tables: { table_name: string; display_name: string; category: string }[]; error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/browser/tables`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { tables: [], error: message };
        }

        const data = await response.json();
        return { tables: data.tables || [], error: null };
    } catch (error) {
        if (error instanceof Error) {
            return { tables: [], error: error.message };
        }
        return { tables: [], error: 'An unknown error occurred.' };
    }
}

export async function fetchTableData(tableName: string): Promise<{
    data: Record<string, unknown>[];
    columns: string[];
    rowCount: number;
    columnCount: number;
    displayName: string;
    error: string | null;
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/browser/data/${encodeURIComponent(tableName)}`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { data: [], columns: [], rowCount: 0, columnCount: 0, displayName: '', error: message };
        }

        const result = await response.json();
        return {
            data: result.data || [],
            columns: result.columns || [],
            rowCount: result.row_count || 0,
            columnCount: result.column_count || 0,
            displayName: result.display_name || tableName,
            error: null,
        };
    } catch (error) {
        if (error instanceof Error) {
            return { data: [], columns: [], rowCount: 0, columnCount: 0, displayName: '', error: error.message };
        }
        return { data: [], columns: [], rowCount: 0, columnCount: 0, displayName: '', error: 'An unknown error occurred.' };
    }
}

// Training API functions
import type { TrainingConfig, TrainingDatasetInfo, CheckpointInfo } from './types';

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
        const checkpoints: CheckpointInfo[] = (result.checkpoints || []).map((name: string) => ({
            name,
        }));
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

export async function resumeTraining(checkpoint: string, additionalEpochs: number): Promise<{
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
                body: JSON.stringify({
                    checkpoint_name: checkpoint,
                    additional_epochs: additionalEpochs,
                }),
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

export async function getTrainingStatus(): Promise<{
    is_training: boolean;
    current_epoch: number;
    total_epochs: number;
    progress: number;
    error: string | null;
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/status`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { is_training: false, current_epoch: 0, total_epochs: 0, progress: 0, error: message };
        }

        const result = await response.json();
        return {
            is_training: result.is_training || false,
            current_epoch: result.current_epoch || 0,
            total_epochs: result.total_epochs || 0,
            progress: result.progress || 0,
            error: null,
        };
    } catch (error) {
        if (error instanceof Error) {
            return { is_training: false, current_epoch: 0, total_epochs: 0, progress: 0, error: error.message };
        }
        return { is_training: false, current_epoch: 0, total_epochs: 0, progress: 0, error: 'An unknown error occurred.' };
    }
}

// Dataset Builder API functions
import type { DatasetBuildConfig, DatasetBuildResult, DatasetFullInfo } from './types';

export async function buildTrainingDataset(config: DatasetBuildConfig): Promise<DatasetBuildResult> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/build-dataset`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            },
            HTTP_TIMEOUT * 2
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { success: false, message };
        }

        const result = await response.json();
        return {
            success: result.success || false,
            message: result.message || 'Dataset built.',
            total_samples: result.total_samples,
            train_samples: result.train_samples,
            validation_samples: result.validation_samples,
        };
    } catch (error) {
        if (error instanceof Error) {
            return { success: false, message: error.message };
        }
        return { success: false, message: 'An unknown error occurred.' };
    }
}

export async function getTrainingDatasetInfo(): Promise<DatasetFullInfo> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/dataset-info`,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            return { available: false };
        }

        const result = await response.json();
        return {
            available: result.available || false,
            created_at: result.created_at,
            sample_size: result.sample_size,
            validation_size: result.validation_size,
            min_measurements: result.min_measurements,
            max_measurements: result.max_measurements,
            smile_sequence_size: result.smile_sequence_size,
            max_pressure: result.max_pressure,
            max_uptake: result.max_uptake,
            total_samples: result.total_samples,
            train_samples: result.train_samples,
            validation_samples: result.validation_samples,
        };
    } catch {
        return { available: false };
    }
}

export async function clearTrainingDataset(): Promise<{ success: boolean; message: string }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/dataset`,
            { method: 'DELETE' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            const message = extractErrorMessage(response, data);
            return { success: false, message };
        }

        const result = await response.json();
        return {
            success: result.status === 'success',
            message: result.message || 'Dataset cleared.',
        };
    } catch (error) {
        if (error instanceof Error) {
            return { success: false, message: error.message };
        }
        return { success: false, message: 'An unknown error occurred.' };
    }
}
