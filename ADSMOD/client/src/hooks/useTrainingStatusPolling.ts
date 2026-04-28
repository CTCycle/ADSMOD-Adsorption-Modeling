import { useCallback, useEffect, useRef, useState } from 'react';
import { getTrainingStatus } from '../services';
import type { TrainingStatus } from '../types';

interface UseTrainingStatusPollingOptions {
    onTrainingEnded?: () => void;
}

interface UseTrainingStatusPollingResult {
    trainingStatus: TrainingStatus;
    checkStatus: () => Promise<void>;
    startPolling: (intervalSeconds?: number) => void;
    appendTrainingLog: (message: string) => void;
    clearTrainingLog: () => void;
}

const DEFAULT_TRAINING_STATUS: TrainingStatus = {
    is_training: false,
    current_epoch: 0,
    total_epochs: 0,
    progress: 0,
    metrics: {},
    history: [],
    log: [],
};

const normalizePollingInterval = (intervalSeconds: number | null | undefined): number | null => {
    if (typeof intervalSeconds !== 'number' || Number.isNaN(intervalSeconds)) {
        return null;
    }
    return intervalSeconds < 0 ? 0 : intervalSeconds;
};

export const useTrainingStatusPolling = ({
    onTrainingEnded,
}: UseTrainingStatusPollingOptions): UseTrainingStatusPollingResult => {
    const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>(DEFAULT_TRAINING_STATUS);
    const pollIntervalRef = useRef<number | null>(null);
    const pollIntervalSecondsRef = useRef<number | null>(null);
    const wasTrainingRef = useRef(false);
    const checkStatusRef = useRef<() => Promise<void>>(async () => undefined);

    const stopPolling = useCallback((): void => {
        if (pollIntervalRef.current) {
            window.clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
        }
    }, []);

    const startPolling = useCallback((intervalSeconds: number = 1.0): void => {
        const normalizedInterval = normalizePollingInterval(intervalSeconds) ?? 1.0;
        if (pollIntervalRef.current) {
            window.clearInterval(pollIntervalRef.current);
        }
        pollIntervalSecondsRef.current = normalizedInterval;
        pollIntervalRef.current = window.setInterval(() => {
            void checkStatusRef.current();
        }, normalizedInterval * 1000);
    }, []);

    const checkStatus = useCallback(async (): Promise<void> => {
        const status = await getTrainingStatus();
        if (status.error) {
            console.error('Failed to poll status:', status.error);
            return;
        }

        const wasTraining = wasTrainingRef.current;
        wasTrainingRef.current = status.is_training;

        setTrainingStatus({
            is_training: status.is_training,
            current_epoch: status.current_epoch,
            total_epochs: status.total_epochs,
            progress: status.progress,
            metrics: status.metrics || {},
            history: status.history || [],
            log: status.log || [],
            poll_interval: status.poll_interval,
        });

        const nextInterval = normalizePollingInterval(status.poll_interval);
        if (nextInterval !== null && pollIntervalSecondsRef.current !== nextInterval) {
            if (status.is_training) {
                startPolling(nextInterval);
            } else {
                pollIntervalSecondsRef.current = nextInterval;
            }
        }

        if (!status.is_training && wasTraining) {
            onTrainingEnded?.();
        }
    }, [onTrainingEnded, startPolling]);

    checkStatusRef.current = checkStatus;

    useEffect(() => {
        if (trainingStatus.is_training) {
            if (!pollIntervalRef.current) {
                const intervalSeconds = trainingStatus.poll_interval ?? pollIntervalSecondsRef.current ?? 1.0;
                startPolling(intervalSeconds);
            }
        } else {
            stopPolling();
        }
    }, [startPolling, stopPolling, trainingStatus.is_training, trainingStatus.poll_interval]);

    useEffect(() => {
        return () => {
            stopPolling();
        };
    }, [stopPolling]);

    const appendTrainingLog = useCallback((message: string): void => {
        setTrainingStatus((previousStatus) => ({
            ...previousStatus,
            log: [...(previousStatus.log || []), message],
        }));
    }, []);

    const clearTrainingLog = useCallback((): void => {
        setTrainingStatus((previousStatus) => ({
            ...previousStatus,
            log: ['Ready to start training...'],
        }));
    }, []);

    return {
        trainingStatus,
        checkStatus,
        startPolling,
        appendTrainingLog,
        clearTrainingLog,
    };
};
