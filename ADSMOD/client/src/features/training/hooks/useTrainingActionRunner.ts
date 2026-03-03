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
                console.error(`Failed to ${actionLabel}:`, result.message);
                alert(`Failed to ${actionLabel}: ${result.message}`);
            } finally {
                setLoading(false);
            }
        },
        [appendLog, setLoading]
    );

    return { runTrainingAction };
};
