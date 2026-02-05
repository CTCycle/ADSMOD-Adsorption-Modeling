import type { FittingPayload, FittingResponse, JobStartResponse, JobStatusResponse } from '../types';
import { API_BASE_URL } from '../constants';
import { fetchWithTimeout, extractErrorMessage, HTTP_TIMEOUT } from './http';
import { pollJobStatus, resolvePollingIntervalMs } from './jobs';

export async function startFittingJob(
    payload: FittingPayload
): Promise<{ jobId: string | null; pollInterval?: number; error: string | null }> {
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
        return { jobId: result.job_id, pollInterval: result.poll_interval, error: null };
    } catch (error) {
        if (error instanceof Error) {
            return { jobId: null, error: error.message };
        }
        return { jobId: null, error: 'An unknown error occurred.' };
    }
}

export async function pollFittingJobUntilComplete(
    jobId: string,
    pollInterval?: number,
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

        await new Promise((resolve) =>
            setTimeout(resolve, resolvePollingIntervalMs(status.poll_interval ?? pollInterval))
        );
    }
}

// Legacy startFitting that uses job polling internally
export async function startFitting(
    payload: FittingPayload,
    onProgress?: (status: JobStatusResponse) => void
): Promise<{ message: string; data: FittingResponse | null }> {
    const { jobId, pollInterval, error } = await startFittingJob(payload);
    if (error || !jobId) {
        return { message: `[ERROR] ${error || 'Failed to start job.'}`, data: null };
    }
    return pollFittingJobUntilComplete(jobId, pollInterval, onProgress);
}
