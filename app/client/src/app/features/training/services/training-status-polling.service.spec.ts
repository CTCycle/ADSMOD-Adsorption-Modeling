import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { TrainingStatusPollingService } from './training-status-polling.service';

describe('TrainingStatusPollingService', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
        TestBed.resetTestingModule();
    });

    it('reports errors without publishing status', async () => {
        fetchMock.mockResolvedValue({
            ok: false,
            status: 500,
            json: async () => ({ detail: 'status failed' }),
        });

        const service = TestBed.runInInjectionContext(() => new TrainingStatusPollingService());
        const onStatus = vi.fn();
        const onError = vi.fn();

        await service.checkStatus(onStatus, onError);

        expect(onStatus).not.toHaveBeenCalled();
        expect(onError).toHaveBeenCalledWith('status failed');
    });

    it('stops polling and calls onTrainingEnded after a training run completes', async () => {
        const service = TestBed.runInInjectionContext(() => new TrainingStatusPollingService());
        const onStatus = vi.fn();
        const onError = vi.fn();
        const onTrainingEnded = vi.fn();

        fetchMock.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                is_training: true,
                current_epoch: 1,
                total_epochs: 4,
                progress: 25,
                metrics: { loss: 0.8 },
                history: [{ epoch: 1, loss: 0.8 }],
                log: ['started'],
                poll_interval: 3,
            }),
        });
        await service.checkStatus(onStatus, onError, onTrainingEnded);

        fetchMock.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                is_training: false,
                current_epoch: 4,
                total_epochs: 4,
                progress: 100,
                metrics: { loss: 0.2 },
                history: [{ epoch: 4, loss: 0.2 }],
                log: ['done'],
                poll_interval: 3,
            }),
        });
        await service.checkStatus(onStatus, onError, onTrainingEnded);

        expect(onError).toHaveBeenLastCalledWith(null);
        expect(onStatus).toHaveBeenCalledTimes(2);
        expect(onTrainingEnded).toHaveBeenCalledTimes(1);
    });

    it('creates fresh default status objects', () => {
        const service = TestBed.runInInjectionContext(() => new TrainingStatusPollingService());

        const first = service.createDefaultStatus();
        const second = service.createDefaultStatus();
        first.log?.push('mutated');

        expect(second).toEqual({
            is_training: false,
            current_epoch: 0,
            total_epochs: 0,
            progress: 0,
            metrics: {},
            history: [],
            log: [],
        });
    });
});
