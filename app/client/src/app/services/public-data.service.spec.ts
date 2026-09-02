import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fetchPublicChemicals, resolvePubChem } from './public-data.service';

describe('public data service', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('sends public-data filters to the backend instead of filtering client-side', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ items: [], pagination: { page: 2, page_size: 25, total: 0 } }),
        });

        await fetchPublicChemicals({
            page: 2,
            page_size: 25,
            q: 'carbon dioxide',
            formula: 'CO2',
            source: 'pubchem',
            molecular_weight_min: 40,
            molecular_weight_max: 50,
        });

        const [requestUrl] = fetchMock.mock.calls[0];
        const url = new URL(String(requestUrl), 'http://localhost');
        expect(url.pathname).toBe('/api/v1/public-data/chemicals');
        expect(url.searchParams.get('page')).toBe('2');
        expect(url.searchParams.get('q')).toBe('carbon dioxide');
        expect(url.searchParams.get('source')).toBe('pubchem');
        expect(url.searchParams.get('molecular_weight_min')).toBe('40');
        expect(url.searchParams.get('molecular_weight_max')).toBe('50');
    });

    it('resolves PubChem records through the normalized public-data endpoint', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ id: 1, name: 'methane' }),
        });

        const result = await resolvePubChem('VNWKTOKETHGBQD-UHFFFAOYSA-N');

        expect(result.error).toBeNull();
        const [requestUrl, requestInit] = fetchMock.mock.calls[0];
        expect(String(requestUrl)).toContain('/api/v1/public-data/chemicals/resolve');
        expect(requestInit.method).toBe('POST');
        expect(JSON.parse(requestInit.body as string)).toEqual({
            query: 'VNWKTOKETHGBQD-UHFFFAOYSA-N',
        });
    });
});
