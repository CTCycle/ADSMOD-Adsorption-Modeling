import type { DatasetBuildConfig, DatasetBuildResult, DatasetFullInfo, JobStartResponse, JobStatusResponse, ProcessedDatasetInfo } from '../types';
import { API_BASE_URL } from '../constants';
import { fetchWithTimeout, extractErrorMessage, HTTP_TIMEOUT } from './http';
import { JOB_POLL_INTERVAL, pollJobStatus } from './jobs';

export async function startTrainingDatasetJob(
    config: DatasetBuildConfig
): Promise<{ jobId: string | null; error: string | null }> {
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

export async function pollTrainingDatasetJobUntilComplete(
    jobId: string,
    onProgress?: (status: JobStatusResponse) => void
): Promise<DatasetBuildResult> {
    while (true) {
        const status = await pollJobStatus('/training', jobId);
        if (!status) {
            return { success: false, message: 'Failed to poll dataset job status.' };
        }

        if (onProgress) {
            onProgress(status);
        }

        if (status.status === 'completed') {
            const result = status.result as DatasetBuildResult | undefined;
            if (!result) {
                return { success: true, message: 'Dataset build completed.' };
            }
            return {
                success: result.success ?? true,
                message: result.message || 'Dataset built.',
                total_samples: result.total_samples,
                train_samples: result.train_samples,
                validation_samples: result.validation_samples,
            };
        }

        if (status.status === 'failed') {
            return { success: false, message: status.error || 'Dataset build failed.' };
        }

        if (status.status === 'cancelled') {
            return { success: false, message: 'Dataset build job was cancelled.' };
        }

        await new Promise((resolve) => setTimeout(resolve, JOB_POLL_INTERVAL));
    }
}

export async function buildTrainingDataset(
    config: DatasetBuildConfig,
    onProgress?: (status: JobStatusResponse) => void
): Promise<DatasetBuildResult> {
    const { jobId, error } = await startTrainingDatasetJob(config);
    if (error || !jobId) {
        return { success: false, message: error || 'Failed to start dataset job.' };
    }
    return pollTrainingDatasetJobUntilComplete(jobId, onProgress);
}

export async function fetchProcessedDatasets(): Promise<{
    datasets: ProcessedDatasetInfo[];
    error: string | null;
}> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/processed-datasets`,
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

export async function getTrainingDatasetInfo(datasetLabel?: string): Promise<DatasetFullInfo> {
    try {
        const url = datasetLabel
            ? `${API_BASE_URL}/training/dataset-info?dataset_label=${encodeURIComponent(datasetLabel)}`
            : `${API_BASE_URL}/training/dataset-info`;

        const response = await fetchWithTimeout(
            url,
            { method: 'GET' },
            HTTP_TIMEOUT
        );

        if (!response.ok) {
            return { available: false };
        }

        const result = await response.json();
        return {
            available: result.available || false,
            dataset_label: result.dataset_label,
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
            smile_vocabulary_size: result.smile_vocabulary_size,
            adsorbent_vocabulary_size: result.adsorbent_vocabulary_size,
            normalization_stats: result.normalization_stats,
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

export async function deleteDataset(
    datasetLabel: string
): Promise<{ success: boolean; message: string }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/training/dataset?dataset_label=${encodeURIComponent(datasetLabel)}`,
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
            message: result.message || 'Dataset deleted.',
        };
    } catch (error) {
        if (error instanceof Error) {
            return { success: false, message: error.message };
        }
        return { success: false, message: 'An unknown error occurred.' };
    }
}



