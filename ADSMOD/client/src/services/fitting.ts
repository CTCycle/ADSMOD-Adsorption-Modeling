import type { FittingPayload, FittingResponse, JobStatusResponse } from '../types';
import { pollJobStatus, resolvePollingIntervalMs, startJob } from './jobs';

export async function startFittingJob(
    payload: FittingPayload
): Promise<{ jobId: string | null; pollInterval?: number; error: string | null }> {
    return startJob('/fitting/run', payload);
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
