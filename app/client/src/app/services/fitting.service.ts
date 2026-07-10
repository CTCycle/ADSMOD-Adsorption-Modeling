import type { FittingPayload, FittingResponse } from '../models/fitting.model';
import type { JobStatusResponse } from '../models/job.model';
import { pollJobUntilTerminal, startJob } from './job.service';
import type { JobStartResult } from './job.service';

export async function startFittingJob(
    payload: FittingPayload
): Promise<JobStartResult> {
    return startJob('/fitting/run', payload);
}

export async function pollFittingJobUntilComplete(
    jobId: string,
    pollInterval?: number,
    onProgress?: (status: JobStatusResponse) => void
): Promise<{ message: string; data: FittingResponse | null }> {
    const status = await pollJobUntilTerminal('/fitting', jobId, pollInterval, onProgress);
    if (!status) {
        return { message: '[ERROR] Failed to poll job status.', data: null };
    }

    if (status.status === 'completed') {
        const result = status.result as FittingResponse | undefined;
        if (!result) {
            return { message: '[INFO] Fitting completed.', data: null };
        }

        const defaultMessage =
            result.status === 'error'
                ? '[ERROR] Fitting finished without any successful model fits.'
                : result.status === 'warning'
                    ? '[WARN] Fitting completed with partial issues.'
                    : '[INFO] Fitting completed successfully.';
        const lines: string[] = [defaultMessage];
        if (typeof result.processed_rows === 'number') {
            lines.push(`Processed experiments: ${result.processed_rows}`);
        }

        return { message: result.summary || lines.join('\n'), data: result };
    }

    if (status.status === 'failed') {
        return { message: `[ERROR] ${status.error || 'Job failed.'}`, data: null };
    }

    return { message: '[INFO] Job was cancelled.', data: null };
}