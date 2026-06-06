import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { pollFittingJobUntilComplete, startFittingJob } from './fitting.service';

describe('fitting.service', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('posts fitting jobs to the expected endpoint', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ job_id: 'fit-1', poll_interval: 2 }),
        });

        await expect(startFittingJob({
            max_iterations: 100,
            optimization_method: 'LSS',
            parameter_bounds: {},
            dataset: { dataset_name: 'set-a', columns: [], records: [] },
        })).resolves.toEqual({
            jobId: 'fit-1',
            pollInterval: 2,
            error: null,
        });

        expect(fetchMock).toHaveBeenCalledWith(
            '/api/fitting/run',
            expect.objectContaining({
                method: 'POST',
                body: expect.any(String),
                signal: expect.any(AbortSignal),
            })
        );
    });

    it('returns the backend summary when the fitting job completes', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                status: 'completed',
                progress: 100,
                result: {
                    summary: 'Best model: Langmuir',
                    processed_rows: 42,
                },
            }),
        });

        await expect(pollFittingJobUntilComplete('fit-42', 0)).resolves.toEqual({
            message: 'Best model: Langmuir',
            data: {
                summary: 'Best model: Langmuir',
                processed_rows: 42,
            },
        });
        expect(fetchMock).toHaveBeenCalledWith(
            '/api/fitting/jobs/fit-42',
            expect.objectContaining({ method: 'GET', signal: expect.any(AbortSignal) })
        );
    });

    it('reports cancelled fitting jobs as informational terminal states', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                status: 'cancelled',
                progress: 25,
            }),
        });

        await expect(pollFittingJobUntilComplete('fit-99', 0)).resolves.toEqual({
            message: '[INFO] Job was cancelled.',
            data: null,
        });
    });
});
