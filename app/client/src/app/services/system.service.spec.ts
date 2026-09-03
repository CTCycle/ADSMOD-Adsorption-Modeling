import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const capabilities = (machineLearning: boolean) => ({
    version: '3.0.0',
    features: {
        datasets: true,
        nist: true,
        fitting: true,
        machine_learning: machineLearning,
        training: machineLearning,
        checkpoints: machineLearning,
    },
});

async function loadService() {
    vi.resetModules();
    return await import('./system.service');
}

describe('system service capability discovery', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
    });

    afterEach(() => {
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
    });

    it('retries capability discovery after a transient failure', async () => {
        fetchMock
            .mockRejectedValueOnce(new Error('backend starting'))
            .mockResolvedValueOnce({
                ok: true,
                json: async () => capabilities(true),
            });
        const { machineLearningAvailable } = await loadService();

        expect(await machineLearningAvailable()).toBe(false);
        expect(await machineLearningAvailable()).toBe(true);
        expect(fetchMock).toHaveBeenCalledTimes(2);
    });

    it('reuses a successful capability response without duplicate requests', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => capabilities(false),
        });
        const { fetchApplicationCapabilities } = await loadService();

        const first = await fetchApplicationCapabilities();
        const second = await fetchApplicationCapabilities();

        expect(first.data?.features.machine_learning).toBe(false);
        expect(second.data).toEqual(first.data);
        expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    it('refreshes a previously cached capability response when requested', async () => {
        fetchMock
            .mockResolvedValueOnce({
                ok: true,
                json: async () => capabilities(false),
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => capabilities(true),
            });
        const { fetchApplicationCapabilities } = await loadService();

        expect((await fetchApplicationCapabilities()).data?.features.machine_learning).toBe(false);
        expect((await fetchApplicationCapabilities(true)).data?.features.machine_learning).toBe(true);
        expect((await fetchApplicationCapabilities()).data?.features.machine_learning).toBe(true);
        expect(fetchMock).toHaveBeenCalledTimes(2);
    });
});
