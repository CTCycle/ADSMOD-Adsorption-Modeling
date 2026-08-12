import { beforeEach, describe, expect, it, vi } from 'vitest';
import { deleteDataset } from './dataset.service';

describe('dataset.service', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
    });

    it('deletes a dataset through the canonical dataset ID route', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            status: 204,
            json: async () => null,
        });

        await expect(deleteDataset(17)).resolves.toEqual({ data: null, error: null });
        expect(fetchMock).toHaveBeenCalledWith(
            '/api/datasets/17',
            expect.objectContaining({ method: 'DELETE', signal: expect.any(AbortSignal) }),
        );
    });
});
