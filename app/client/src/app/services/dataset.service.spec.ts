import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fetchDatasetByName, fetchDatasetNames, loadDataset } from './dataset.service';

describe('dataset.service', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
    });

    it('returns dataset names from the expected endpoint', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ names: ['set_a', 'set_b'] }),
        });

        await expect(fetchDatasetNames()).resolves.toEqual({
            names: ['set_a', 'set_b'],
            error: null,
        });
        expect(fetchMock).toHaveBeenCalledWith(
            '/api/datasets/names',
            expect.objectContaining({ method: 'GET', signal: expect.any(AbortSignal) })
        );
    });

    it('normalizes dataset payloads returned by dataset lookup', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({
                status: 'success',
                summary: 'Loaded',
                dataset: {
                    dataset_name: 'zeolite-screen',
                    columns: undefined,
                    records: undefined,
                },
            }),
        });

        await expect(fetchDatasetByName('zeolite-screen')).resolves.toEqual({
            dataset: {
                dataset_name: 'zeolite-screen',
                columns: [],
                records: [],
            },
            summary: 'Loaded',
            error: null,
        });
        expect(fetchMock).toHaveBeenCalledWith(
            '/api/datasets/by-name/zeolite-screen',
            expect.objectContaining({ method: 'GET', signal: expect.any(AbortSignal) })
        );
    });

    it('returns formatted backend errors for failed uploads', async () => {
        fetchMock.mockResolvedValue({
            ok: false,
            status: 400,
            json: async () => ({ detail: 'invalid file' }),
        });

        const result = await loadDataset(new File(['a,b\n1,2'], 'sample.csv', { type: 'text/csv' }));

        expect(result).toEqual({
            dataset: null,
            message: '[ERROR] invalid file',
        });
        expect(fetchMock).toHaveBeenCalledWith(
            '/api/datasets/load',
            expect.objectContaining({
                method: 'POST',
                body: expect.any(FormData),
                signal: expect.any(AbortSignal),
            })
        );
    });
});
