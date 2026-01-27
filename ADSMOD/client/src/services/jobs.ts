import type { JobStatusResponse } from '../types';
import { API_BASE_URL } from '../constants';
import { fetchWithTimeout, HTTP_TIMEOUT } from './http';

export const JOB_POLL_INTERVAL = 1000; // 1 second

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
