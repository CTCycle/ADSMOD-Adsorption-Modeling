import type {
    DatasetPayload,
    JobStartResponse,
    JobStatusResponse,
    NISTFetchRequest,
    NISTPropertiesRequest,
    NISTStatusResponse,
} from '../types';
import { API_BASE_URL } from '../constants';
import { fetchWithTimeout, extractErrorMessage, HTTP_TIMEOUT } from './http';
import { JOB_POLL_INTERVAL, pollJobStatus } from './jobs';

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
