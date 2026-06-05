import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { TrainingActionRunnerService } from './training-action-runner.service';

describe('TrainingActionRunnerService', () => {
    let service: TrainingActionRunnerService;

    beforeEach(() => {
        TestBed.resetTestingModule();
        service = TestBed.inject(TrainingActionRunnerService);
    });

    it('appends success messages and calls onSuccess for successful actions', async () => {
        const setLoading = vi.fn();
        const appendLog = vi.fn();
        const onSuccess = vi.fn();

        await service.runTrainingAction({
            action: async () => ({ status: 'started' as const, message: 'Training started.', poll_interval: 2 }),
            successStatus: 'started',
            actionLabel: 'start training',
            setLoading,
            appendLog,
            onSuccess,
        });

        expect(setLoading).toHaveBeenNthCalledWith(1, true);
        expect(appendLog).toHaveBeenCalledWith('Training started.');
        expect(onSuccess).toHaveBeenCalledWith({
            status: 'started',
            message: 'Training started.',
            poll_interval: 2,
        });
        expect(setLoading).toHaveBeenLastCalledWith(false);
    });

    it('logs formatted errors when the action returns an error status', async () => {
        const setLoading = vi.fn();
        const appendLog = vi.fn();

        await service.runTrainingAction({
            action: async () => ({ status: 'error' as const, message: 'missing dataset' }),
            successStatus: 'started',
            actionLabel: 'start training',
            setLoading,
            appendLog,
        });

        expect(appendLog).toHaveBeenCalledWith('[ERROR] Failed to start training: missing dataset');
        expect(setLoading).toHaveBeenLastCalledWith(false);
    });

    it('logs thrown exceptions with a formatted message', async () => {
        const setLoading = vi.fn();
        const appendLog = vi.fn();

        await service.runTrainingAction({
            action: async () => {
                throw new Error('boom');
            },
            successStatus: 'stopped',
            actionLabel: 'stop training',
            setLoading,
            appendLog,
        });

        expect(appendLog).toHaveBeenCalledWith('[ERROR] Failed to stop training: boom');
        expect(setLoading).toHaveBeenLastCalledWith(false);
    });
});
