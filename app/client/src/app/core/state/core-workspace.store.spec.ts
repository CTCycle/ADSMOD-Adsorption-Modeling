import { beforeEach, describe, expect, it, vi } from 'vitest';
import { CoreWorkspaceStore } from './core-workspace.store';

describe('CoreWorkspaceStore', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
    });

    it('loads available datasets during construction', async () => {
        fetchMock.mockResolvedValueOnce({
            ok: true,
            json: async () => ({ datasets: [{ name: 'dataset-a' }, { name: 'dataset-b' }] }),
        });
        fetchMock.mockResolvedValueOnce({
            ok: true,
            json: async () => ({ datasets: [{ name: 'dataset-a' }, { name: 'dataset-b' }] }),
        });

        const store = new CoreWorkspaceStore();
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(store.availableDatasets()).toEqual(['dataset-a', 'dataset-b']);
        expect(fetchMock).toHaveBeenCalledWith(
            '/api/datasets',
            expect.objectContaining({ method: 'GET', signal: expect.any(AbortSignal) })
        );
    });

    it('requires at least one enabled model before starting fitting', async () => {
        fetchMock.mockResolvedValueOnce({
            ok: true,
            json: async () => ({ datasets: [] }),
        });

        const store = new CoreWorkspaceStore();
        await new Promise((resolve) => setTimeout(resolve, 0));
        fetchMock.mockClear();

        Object.keys(store.modelStates()).forEach((modelName) => {
            store.setModelEnabled(modelName, false);
        });
        store.setSelectedDataset('loaded-set');

        await store.startFitting();

        expect(store.fittingStatus()).toBe('[ERROR] Select at least one model.');
        expect(fetchMock).not.toHaveBeenCalled();
    });

    it('swaps inverted parameter bounds and sends midpoint initials to the fitting endpoint', async () => {
        fetchMock
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ datasets: [] }),
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ job_id: 'job-42', poll_interval: 0 }),
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    status: 'completed',
                    progress: 100,
                    poll_interval: 0,
                    result: { summary: 'Fitting finished.' },
                }),
            });

        const store = new CoreWorkspaceStore();
        await new Promise((resolve) => setTimeout(resolve, 0));
        fetchMock.mockClear();

        store.setSelectedDataset('loaded-set');
        store.setModelParameters('Langmuir', {
            k: { min: 4, max: 2 },
            qsat: { min: 0, max: 10 },
        });
        Object.keys(store.modelStates()).forEach((modelName) => {
            if (modelName !== 'Langmuir') {
                store.setModelEnabled(modelName, false);
            }
        });

        await store.startFitting();

        const fittingRequest = fetchMock.mock.calls.find(([url]) => url === '/api/fitting/run') as [string, RequestInit] | undefined;
        expect(fittingRequest).toBeDefined();
        const [, requestInit] = fittingRequest!;
        const payload = JSON.parse(String(requestInit.body)) as {
            parameter_bounds: Record<string, { min: Record<string, number>; max: Record<string, number>; initial: Record<string, number> }>;
        };

        expect(payload.parameter_bounds['Langmuir']).toEqual({
            min: { k: 2, qsat: 0 },
            max: { k: 4, qsat: 10 },
            initial: { k: 3, qsat: 5 },
        });
        expect(store.fittingStatus()).toBe('Fitting finished.');
    });
});
