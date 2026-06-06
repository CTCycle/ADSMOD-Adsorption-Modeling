import { describe, expect, it, vi } from 'vitest';
import { TrainingActionRunnerService } from './training-action-runner.service';

describe('TrainingActionRunnerService', () => {
    it('awaits an asynchronous success handler before clearing loading state', async () => {
        const service = new TrainingActionRunnerService();
        const events: string[] = [];
        const action = vi.fn(async () => ({
            status: 'started' as const,
            message: 'Training started.',
        }));
        const setLoading = vi.fn((loading: boolean) => {
            events.push(`loading:${loading}`);
        });
        const appendLog = vi.fn((message: string) => {
            events.push(`log:${message}`);
        });
        const onSuccess = vi.fn(async () => {
            events.push('success:start');
            await Promise.resolve();
            events.push('success:end');
        });

        await service.runTrainingAction({
            action,
            successStatus: 'started',
            actionLabel: 'start training',
            setLoading,
            appendLog,
            onSuccess,
        });

        expect(action).toHaveBeenCalledTimes(1);
        expect(onSuccess).toHaveBeenCalledTimes(1);
        expect(setLoading).toHaveBeenCalledTimes(2);
        expect(events).toEqual(['loading:true', 'log:Training started.', 'success:start', 'success:end', 'loading:false']);
    });
});