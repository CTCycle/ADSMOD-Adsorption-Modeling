// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    fetchPublicOverview,
    checkPublicSourceHealth,
    fetchPublicDataSyncFailures,
    fetchPublicSources,
    resolvePubChem,
    cancelPublicDataSyncRun,
    importCOD,
    searchNistIsotherms,
    fetchNistIsotherms,
    fetchBulkStructures,
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

    it('searches the NIST index and queues a targeted fetch of the chosen isotherms', async () => {
        fetchMock.mockResolvedValue({ ok: true, json: async () => ({ items: [], total: 0 }) });

        await searchNistIsotherms({ adsorbate: 'CO2', temperature_min: null, limit: 10, offset: 20 });
        await fetchNistIsotherms(['iso-1', 'iso-2']);
        await fetchBulkStructures(0.5);

        expect(String(fetchMock.mock.calls[0][0])).toBe('/api/v1/public-data/sources/nist/search?adsorbate=CO2&limit=10&offset=20');
        expect(JSON.parse(fetchMock.mock.calls[1][1].body as string)).toEqual({
            source_key: 'nist',
            operation: 'nist_category',
            category: 'experiments',
            identifiers: ['iso-1', 'iso-2'],
        });
        expect(JSON.parse(fetchMock.mock.calls[2][1].body as string)).toEqual({
            source_key: 'cod',
            operation: 'cod_import',
            fraction: 0.5,
        });
    });
});
