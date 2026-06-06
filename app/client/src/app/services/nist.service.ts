import { API_BASE_URL } from '../core/config/api-base-url';
import type { DatasetPayload } from '../models/dataset.model';
import type { JobStatusResponse } from '../models/job.model';
import type {
    NISTCategoryFetchRequest,
    NISTCategoryKey,
    NISTCategoryPingResponse,
    NISTCategoryStatusResponse,
    NISTFetchRequest,
    NISTPropertiesRequest,
    NISTStatusResponse,
} from '../models/nist.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';
import { pollJobStatus, resolvePollingIntervalMs, startJob } from './job.service';

async function startCategoryJob(
    endpoint: string,
    payload?: unknown
): Promise<{ jobId: string | null; pollInterval?: number; error: string | null }> {
    return startJob(endpoint, payload || {});
}

export async function fetchNistDataForFitting(): Promise<{ dataset: DatasetPayload | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/fitting/nist-dataset`, { method: 'GET' }, HTTP_TIMEOUT);
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { dataset: null, error: extractErrorMessage(response, data) };
        }

        const result = (await response.json()) as { status?: string; detail?: string; message?: string; dataset?: DatasetPayload };
        if (result.status !== 'success') {
            return { dataset: null, error: result.detail || result.message || 'Failed to load NIST data.' };
        }
        if (!result.dataset || typeof result.dataset !== 'object') {
            return { dataset: null, error: 'NIST dataset response did not include dataset records.' };
        }

        return {
            dataset: {
                dataset_name: result.dataset.dataset_name,
                columns: Array.isArray(result.dataset.columns) ? result.dataset.columns : [],
                records: Array.isArray(result.dataset.records) ? result.dataset.records : [],
            },
            error: null,
        };
    } catch (error) {
        return { dataset: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}

export async function startNistFetchJob(
    payload: NISTFetchRequest
): Promise<{ jobId: string | null; pollInterval?: number; error: string | null }> {
    return startJob('/nist/fetch', payload);
}

export async function startNistPropertiesJob(
    payload: NISTPropertiesRequest
): Promise<{ jobId: string | null; pollInterval?: number; error: string | null }> {
    return startJob('/nist/properties', payload);
}

export async function pollNistJobUntilComplete(
    jobId: string,
    pollInterval?: number,
    onProgress?: (status: JobStatusResponse) => void
): Promise<{ result: Record<string, unknown> | null; error: string | null }> {
    while (true) {
        const status = await pollJobStatus('/nist', jobId);
        if (!status) {
            return { result: null, error: 'Failed to poll job status.' };
        }

        onProgress?.(status);

        if (status.status === 'completed') {
            return { result: status.result || null, error: null };
        }
        if (status.status === 'failed') {
            return { result: null, error: status.error || 'Job failed.' };
        }
        if (status.status === 'cancelled') {
            return { result: null, error: 'Job was cancelled.' };
        }

        await new Promise((resolve) => setTimeout(resolve, resolvePollingIntervalMs(status.poll_interval ?? pollInterval)));
    }
}

export async function fetchNistStatus(): Promise<{ data: NISTStatusResponse | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/nist/status`, { method: 'GET' }, HTTP_TIMEOUT);
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { data: null, error: extractErrorMessage(response, data) };
        }

        const result = (await response.json()) as NISTStatusResponse;
        return result.status === 'success'
            ? { data: result, error: null }
            : { data: result, error: result.detail || result.message || 'Failed to load NIST status.' };
    } catch (error) {
        return { data: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}

export async function fetchNistCategoryStatus(): Promise<{ data: NISTCategoryStatusResponse | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/nist/categories/status`, { method: 'GET' }, HTTP_TIMEOUT);
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { data: null, error: extractErrorMessage(response, data) };
        }

        const result = (await response.json()) as NISTCategoryStatusResponse;
        return result.status === 'success'
            ? { data: result, error: null }
            : { data: result, error: result.detail || result.message || 'Failed to load NIST category status.' };
    } catch (error) {
        return { data: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}

export async function pingNistCategoryServer(category: NISTCategoryKey): Promise<{ data: NISTCategoryPingResponse | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/nist/categories/${category}/ping`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            },
            HTTP_TIMEOUT
        );
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { data: null, error: extractErrorMessage(response, data) };
        }

        const result = (await response.json()) as NISTCategoryPingResponse;
        return result.status === 'success'
            ? { data: result, error: null }
            : { data: result, error: result.detail || result.message || 'Failed to ping NIST server.' };
    } catch (error) {
        return { data: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}

export async function startNistCategoryIndexJob(
    category: NISTCategoryKey
): Promise<{ jobId: string | null; pollInterval?: number; error: string | null }> {
    return startCategoryJob(`/nist/categories/${category}/index`);
}

export async function startNistCategoryFetchJob(
    category: NISTCategoryKey,
    payload: NISTCategoryFetchRequest
): Promise<{ jobId: string | null; pollInterval?: number; error: string | null }> {
    return startCategoryJob(`/nist/categories/${category}/fetch`, payload);
}

export async function startNistCategoryEnrichJob(
    category: NISTCategoryKey
): Promise<{ jobId: string | null; pollInterval?: number; error: string | null }> {
    return startCategoryJob(`/nist/categories/${category}/enrich`);
}
