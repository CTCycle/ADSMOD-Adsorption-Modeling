import { API_BASE_URL } from '../core/config/api-base-url';
import type { DatasetPayload, DatasetResponse } from '../models/dataset.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';

function normalizeDatasetPayload(dataset: DatasetPayload): DatasetPayload {
    return {
        dataset_name: dataset.dataset_name,
        columns: Array.isArray(dataset.columns) ? dataset.columns : [],
        records: Array.isArray(dataset.records) ? dataset.records : [],
    };
}

export async function loadDataset(file: File): Promise<{ dataset: DatasetPayload | null; message: string }> {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/datasets/load`, { method: 'POST', body: formData }, HTTP_TIMEOUT);
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { dataset: null, message: `[ERROR] ${extractErrorMessage(response, data)}` };
        }

        const data = (await response.json()) as DatasetResponse;
        if (data.status !== 'success') {
            return { dataset: null, message: `[ERROR] ${data.detail || 'Failed to load dataset.'}` };
        }
        if (!data.dataset) {
            return { dataset: null, message: '[ERROR] Backend returned an invalid dataset payload.' };
        }

        return {
            dataset: normalizeDatasetPayload(data.dataset),
            message: data.summary || '[INFO] Dataset loaded successfully.',
        };
    } catch (error) {
        return {
            dataset: null,
            message: `[ERROR] Failed to reach ADSMOD backend: ${error instanceof Error ? error.message : 'Unknown error.'}`,
        };
    }
}

export async function fetchDatasetNames(): Promise<{ names: string[]; error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/datasets/names`, { method: 'GET' }, HTTP_TIMEOUT);
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { names: [], error: extractErrorMessage(response, data) };
        }

        const result = (await response.json()) as { names?: string[] };
        return { names: result.names || [], error: null };
    } catch (error) {
        return { names: [], error: error instanceof Error ? error.message : 'An unknown error occurred.' };
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
            return { dataset: null, summary: null, error: extractErrorMessage(response, data) };
        }

        const data = (await response.json()) as DatasetResponse;
        if (data.status !== 'success') {
            return { dataset: null, summary: null, error: data.detail || data.message || 'Failed to load dataset.' };
        }

        return {
            dataset: data.dataset ? normalizeDatasetPayload(data.dataset) : null,
            summary: data.summary || null,
            error: null,
        };
    } catch (error) {
        return { dataset: null, summary: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}
