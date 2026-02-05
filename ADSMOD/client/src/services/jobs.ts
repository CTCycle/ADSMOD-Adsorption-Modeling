import type { JobStatusResponse } from '../types';
import { API_BASE_URL } from '../constants';
import { fetchWithTimeout, HTTP_TIMEOUT } from './http';

const DEFAULT_JOB_POLL_INTERVAL_MS = 1000;

export const normalizePollingIntervalSeconds = (
    intervalSeconds: number | null | undefined
): number | null => {
    if (typeof intervalSeconds !== 'number' || Number.isNaN(intervalSeconds)) {
        return null;
    }
    return intervalSeconds < 0 ? 0 : intervalSeconds;
};

export const resolvePollingIntervalMs = (
    intervalSeconds: number | null | undefined
): number => {
    const normalizedSeconds = normalizePollingIntervalSeconds(intervalSeconds);
    if (normalizedSeconds === null) {
        return DEFAULT_JOB_POLL_INTERVAL_MS;
    }
    return normalizedSeconds * 1000;
};

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
