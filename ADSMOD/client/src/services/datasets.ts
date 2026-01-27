import type { DatasetPayload, DatasetResponse } from '../types';
import { API_BASE_URL } from '../constants';
import { fetchWithTimeout, extractErrorMessage, HTTP_TIMEOUT } from './http';

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
