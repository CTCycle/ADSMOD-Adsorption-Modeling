import { Injectable } from '@angular/core';

type TrainingActionResult<TSuccessStatus extends string> = {
    status: TSuccessStatus | 'error';
    message: string;
    poll_interval?: number;
};

interface RunTrainingActionOptions<TSuccessStatus extends string> {
    action: () => Promise<TrainingActionResult<TSuccessStatus>>;
    successStatus: TSuccessStatus;
    actionLabel: string;
    setLoading: (loading: boolean) => void;
    appendLog: (message: string) => void;
    onSuccess?: (result: TrainingActionResult<TSuccessStatus>) => Promise<void> | void;
}

@Injectable({ providedIn: 'root' })
export class TrainingActionRunnerService {
    async runTrainingAction<TSuccessStatus extends string>({
        action,
        successStatus,
        actionLabel,
        setLoading,
        appendLog,
        onSuccess,
    }: RunTrainingActionOptions<TSuccessStatus>): Promise<void> {
        setLoading(true);
        try {
            const result = await action();
            if (result.status === successStatus) {
                appendLog(result.message);
                await onSuccess?.(result);
                return;
            }

            const message = result.message || `Failed to ${actionLabel}.`;
            console.error(`Failed to ${actionLabel}:`, message);
            appendLog(`[ERROR] Failed to ${actionLabel}: ${message}`);
        } catch (error) {
            const message = error instanceof Error ? error.message : 'An unknown error occurred.';
            console.error(`Unexpected error while trying to ${actionLabel}:`, error);
            appendLog(`[ERROR] Failed to ${actionLabel}: ${message}`);
        } finally {
            setLoading(false);
        }
    }
}
