// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    fetchPublicChemicals,
    fetchPublicOverview,
    checkPublicSourceHealth,
    fetchPublicDataSyncFailures,
    fetchPublicSources,
    resolvePubChem,
    cancelPublicDataSyncRun,
    importCOD,
    setStructureMaterial,
} from './public-data.service';

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

    it('queues PubChem resolution as a durable sync run', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ id: 'run-1', source_key: 'pubchem', operation: 'pubchem_resolve', status: 'queued' }),
        });

        const result = await resolvePubChem('VNWKTOKETHGBQD-UHFFFAOYSA-N');

        expect(result.error).toBeNull();
        const [requestUrl, requestInit] = fetchMock.mock.calls[0];
        expect(String(requestUrl)).toContain('/api/v1/public-data/sync-runs');
        expect(requestInit.method).toBe('POST');
        expect(JSON.parse(requestInit.body as string)).toEqual({
            source_key: 'pubchem',
            operation: 'pubchem_resolve',
            query: 'VNWKTOKETHGBQD-UHFFFAOYSA-N',
        });
    });

    it('loads all local Overview totals from one endpoint', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ adsorption_records: 40, materials: 12, chemicals: 8, structures: 3, active_sync_runs: 1, failed_sync_runs: 0, latest_successful_sync_at: null }),
        });

        const result = await fetchPublicOverview();

        expect(result.data).toEqual({ adsorption_records: 40, materials: 12, chemicals: 8, structures: 3, active_sync_runs: 1, failed_sync_runs: 0, latest_successful_sync_at: null });
        expect(result.error).toBeNull();
        expect(fetchMock).toHaveBeenCalledTimes(1);
        expect(String(fetchMock.mock.calls[0][0])).toContain('/api/v1/public-data/overview');
    });

    it('loads source registration without requesting provider health', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ sources: [] }),
        });

        const result = await fetchPublicSources();

        expect(result.data).toEqual({ sources: [] });
        expect(fetchMock).toHaveBeenCalledTimes(1);
        expect(String(fetchMock.mock.calls[0][0])).toBe('/api/v1/public-data/sources');
    });

    it('queues provider health checks without waiting for the provider', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ id: 'health-1', source_key: 'nist', operation: 'health_check', status: 'queued' }),
        });

        const result = await checkPublicSourceHealth('nist');

        expect(result.data?.source_key).toBe('nist');
        expect(fetchMock.mock.calls[0][1].method).toBe('POST');
        expect(fetchMock).toHaveBeenCalledTimes(1);
        expect(String(fetchMock.mock.calls[0][0])).toBe('/api/v1/public-data/sources/nist/health-check');
    });

    it('posts cooperative cancellation to the durable run action', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ id: 'run-1', status: 'cancelled' }),
        });

        const result = await cancelPublicDataSyncRun('run-1');

        expect(result.data?.status).toBe('cancelled');
        expect(fetchMock.mock.calls[0][1].method).toBe('POST');
        expect(String(fetchMock.mock.calls[0][0])).toBe('/api/v1/public-data/sync-runs/run-1/cancel');
    });

    it('loads paginated record failure details for a durable run', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => ({ failures: [], total: 0 }),
        });

        const result = await fetchPublicDataSyncFailures('run-1');

        expect(result.data).toEqual({ failures: [], total: 0 });
        expect(String(fetchMock.mock.calls[0][0])).toBe('/api/v1/public-data/sync-runs/run-1/failures?limit=100&offset=0');
    });

    it('imports one or many COD records as a single durable run with an optional material link', async () => {
        fetchMock.mockResolvedValue({ ok: true, json: async () => ({ id: 'run-2', status: 'queued' }) });

        await importCOD(['1000000']);
        await importCOD(['1000000', '1000001'], 4);

        expect(JSON.parse(fetchMock.mock.calls[0][1].body as string)).toEqual({
            source_key: 'cod',
            operation: 'cod_import',
            cod_id: '1000000',
        });
        expect(JSON.parse(fetchMock.mock.calls[1][1].body as string)).toEqual({
            source_key: 'cod',
            operation: 'cod_import',
            cod_ids: ['1000000', '1000001'],
            adsorbent_id: 4,
        });
    });

    it('links and unlinks a structure through the explicit material endpoint', async () => {
        fetchMock.mockResolvedValue({ ok: true, json: async () => ({ id: 3, material_id: null }) });

        await setStructureMaterial(3, null);

        const [requestUrl, requestInit] = fetchMock.mock.calls[0];
        expect(String(requestUrl)).toBe('/api/v1/public-data/structures/3/material');
        expect(requestInit.method).toBe('PUT');
        expect(JSON.parse(requestInit.body as string)).toEqual({ adsorbent_id: null });
    });
});
