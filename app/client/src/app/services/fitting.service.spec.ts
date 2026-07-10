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
            dataset: { source: 'uploaded', dataset_name: 'set-a' },
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
                    status: 'success',
                    summary: 'Best model: Langmuir',
                    processed_rows: 42,
                },
            }),
        });

        await expect(pollFittingJobUntilComplete('fit-42', 0)).resolves.toEqual({
            message: 'Best model: Langmuir',
            data: {
                status: 'success',
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

    it('falls back to a warning message when a completed job reports partial issues', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                status: 'completed',
                progress: 100,
                result: {
                    status: 'warning',
                    processed_rows: 2,
                },
            }),
        });

        await expect(pollFittingJobUntilComplete('fit-warning', 0)).resolves.toEqual({
            message: '[WARN] Fitting completed with partial issues.\nProcessed experiments: 2',
            data: {
                status: 'warning',
                processed_rows: 2,
            },
        });
    });

    it('falls back to an error message when no model fits succeed', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                status: 'completed',
                progress: 100,
                result: {
                    status: 'error',
                    processed_rows: 0,
                },
            }),
        });

        await expect(pollFittingJobUntilComplete('fit-error', 0)).resolves.toEqual({
            message: '[ERROR] Fitting finished without any successful model fits.\nProcessed experiments: 0',
            data: {
                status: 'error',
                processed_rows: 0,
            },
        });
    });
});
