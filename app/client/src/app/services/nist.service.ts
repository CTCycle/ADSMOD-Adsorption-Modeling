import { API_BASE_URL } from '../core/config/api-base-url';
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
import { pollJobUntilTerminal, startJob } from './job.service';
import type { JobStartResult } from './job.service';

async function startCategoryJob(
    endpoint: string,
    payload?: unknown
): Promise<JobStartResult> {
    return startJob(endpoint, payload || {});
}

export async function startNistFetchJob(
    payload: NISTFetchRequest
): Promise<JobStartResult> {
    return startJob('/nist/fetch', payload);
}

export async function startNistPropertiesJob(
    payload: NISTPropertiesRequest
): Promise<JobStartResult> {
    return startJob('/nist/properties', payload);
}

export async function pollNistJobUntilComplete(
    jobId: string,
    pollInterval?: number,
    onProgress?: (status: JobStatusResponse) => void
): Promise<{ result: Record<string, unknown> | null; error: string | null }> {
    const pollResult = await pollJobUntilTerminal('/nist', jobId, pollInterval, onProgress);
    if (pollResult.error || !pollResult.data) {
        return { result: null, error: pollResult.error || 'Job status response was empty.' };
    }
    const status = pollResult.data;

    if (status.status === 'completed') {
        return { result: status.result || null, error: null };
    }
    if (status.status === 'failed') {
        return { result: null, error: status.error || 'Job failed.' };
    }

    return { result: null, error: 'Job was cancelled.' };
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
): Promise<JobStartResult> {
    return startCategoryJob(`/nist/categories/${category}/index`);
}

export async function startNistCategoryFetchJob(
    category: NISTCategoryKey,
    payload: NISTCategoryFetchRequest
): Promise<JobStartResult> {
    return startCategoryJob(`/nist/categories/${category}/fetch`, payload);
}

export async function startNistCategoryEnrichJob(
    category: NISTCategoryKey
): Promise<JobStartResult> {
    return startCategoryJob(`/nist/categories/${category}/enrich`);
}
