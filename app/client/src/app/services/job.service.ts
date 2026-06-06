import type { JobStartResponse, JobStatusResponse } from '../models/job.model';
import { API_BASE_URL } from '../core/config/api-base-url';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';

const DEFAULT_JOB_POLL_INTERVAL_MS = 1000;

export type JobStartResult = {
    jobId: string | null;
    pollInterval?: number;
    error: string | null;
};

export const normalizePollingIntervalSeconds = (intervalSeconds: number | null | undefined): number | null => {
    if (typeof intervalSeconds !== 'number' || Number.isNaN(intervalSeconds)) {
        return null;
    }

    return intervalSeconds < 0 ? 0 : intervalSeconds;
};

export const resolvePollingIntervalMs = (intervalSeconds: number | null | undefined): number => {
    const normalizedSeconds = normalizePollingIntervalSeconds(intervalSeconds);
    if (normalizedSeconds === null) {
        return DEFAULT_JOB_POLL_INTERVAL_MS;
    }

    return normalizedSeconds * 1000;
};

export async function pollJobStatus(endpoint: string, jobId: string): Promise<JobStatusResponse | null> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}${endpoint}/jobs/${jobId}`, { method: 'GET' }, HTTP_TIMEOUT);
        if (!response.ok) {
            return null;
        }

        return (await response.json()) as JobStatusResponse;
    } catch {
        return null;
    }
}

export async function startJob(
    endpoint: string,
    payload: unknown = {},
    timeout: number = HTTP_TIMEOUT
): Promise<JobStartResult> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}${endpoint}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            },
            timeout
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { jobId: null, error: extractErrorMessage(response, data) };
        }

        const result = (await response.json()) as JobStartResponse;
        return { jobId: result.job_id, pollInterval: result.poll_interval, error: null };
    } catch (error) {
        return { jobId: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}
