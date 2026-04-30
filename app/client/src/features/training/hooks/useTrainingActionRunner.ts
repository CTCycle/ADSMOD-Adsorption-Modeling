import { useCallback } from 'react';

type TrainingActionResult<TSuccessStatus extends string> = {
    status: TSuccessStatus | 'error';
    message: string;
    poll_interval?: number;
};

interface RunTrainingActionOptions<TSuccessStatus extends string> {
    action: () => Promise<TrainingActionResult<TSuccessStatus>>;
    successStatus: TSuccessStatus;
    actionLabel: string;
    onSuccess?: (result: TrainingActionResult<TSuccessStatus>) => void;
}

interface UseTrainingActionRunnerParams {
    setLoading: (loading: boolean) => void;
    appendLog: (message: string) => void;
}

export interface UseTrainingActionRunnerResult {
    runTrainingAction: <TSuccessStatus extends string>(
        options: RunTrainingActionOptions<TSuccessStatus>
    ) => Promise<void>;
}

export const useTrainingActionRunner = ({
    setLoading,
    appendLog,
}: UseTrainingActionRunnerParams): UseTrainingActionRunnerResult => {
    const runTrainingAction = useCallback(
        async <TSuccessStatus extends string>({
            action,
            successStatus,
            actionLabel,
            onSuccess,
        }: RunTrainingActionOptions<TSuccessStatus>): Promise<void> => {
            setLoading(true);
            try {
                const result = await action();
                if (result.status === successStatus) {
                    appendLog(result.message);
                    onSuccess?.(result);
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
        },
        [appendLog, setLoading]
    );

    return { runTrainingAction };
};
