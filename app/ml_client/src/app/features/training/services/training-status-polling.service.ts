import { DestroyRef, Injectable, inject } from '@angular/core';
import { Subscription, timer } from 'rxjs';
import { getTrainingStatus } from '../../../services/training.service';
import type { TrainingStatus } from '../../../models/training.model';

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

@Injectable({ providedIn: 'root' })
export class TrainingStatusPollingService {
    private readonly destroyRef = inject(DestroyRef);
    private pollSubscription: Subscription | null = null;
    private pollIntervalSeconds: number | null = null;
    private wasTraining = false;

    constructor() {
        this.destroyRef.onDestroy(() => this.stopPolling());
    }

    startPolling(
        intervalSeconds: number | undefined,
        onStatus: (status: TrainingStatus) => void,
        onError: (error: string | null) => void,
        onTrainingEnded?: () => void
    ): void {
        const normalizedInterval = normalizePollingInterval(intervalSeconds) ?? 1.0;
        this.stopPolling();
        this.pollIntervalSeconds = normalizedInterval;
        this.pollSubscription = timer(0, normalizedInterval * 1000).subscribe(() => {
            void this.checkStatus(onStatus, onError, onTrainingEnded);
        });
    }

    stopPolling(): void {
        this.pollSubscription?.unsubscribe();
        this.pollSubscription = null;
    }

    async checkStatus(
        onStatus: (status: TrainingStatus) => void,
        onError: (error: string | null) => void,
        onTrainingEnded?: () => void
    ): Promise<void> {
        const status = await getTrainingStatus();
        if (status.error) {
            onError(status.error);
            return;
        }

        onError(null);
        const wasTraining = this.wasTraining;
        this.wasTraining = status.is_training;
        const nextStatus: TrainingStatus = {
            is_training: status.is_training,
            current_epoch: status.current_epoch,
            total_epochs: status.total_epochs,
            progress: status.progress,
            metrics: status.metrics || {},
            history: status.history || [],
            log: status.log || [],
            poll_interval: status.poll_interval,
        };
        onStatus(nextStatus);

        const nextInterval = normalizePollingInterval(status.poll_interval);
        if (status.is_training) {
            if (nextInterval !== null && nextInterval !== this.pollIntervalSeconds) {
                this.startPolling(nextInterval, onStatus, onError, onTrainingEnded);
                return;
            }
        } else {
            this.stopPolling();
            if (wasTraining) {
                onTrainingEnded?.();
            }
        }
    }

    createDefaultStatus(): TrainingStatus {
        return { ...DEFAULT_TRAINING_STATUS, metrics: {}, history: [], log: [] };
    }
}
