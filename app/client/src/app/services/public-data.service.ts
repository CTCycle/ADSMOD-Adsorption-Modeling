// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { API_BASE_URL } from '../core/config/api-base-url';
import type {
    ChemicalLookupResponse,
    CODSearchResponse,
    NISTSearchResponse,
    NISTCollectionListResponse,
    NISTEditedExperimentsResponse,
    PublicDataOverviewResponse,
    PublicDataSyncFailureListResponse,
    PublicDataSyncRequest,
    PublicDataSyncRun,
    PublicDataSyncRunListResponse,
    PublicSourceListResponse,
    StructurePageResponse,
    StructureRecordView,
} from '../models/public-data.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';
import { toErrorMessage } from '../shared/utils/error-message';

export interface ApiResult<T> {
    data: T | null;
    error: string | null;
}

function queryString(params: Record<string, string | number | boolean | null | undefined>): string {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
        if (value !== null && value !== undefined && value !== '') {
            query.set(key, String(value));
        }
    }
    const encoded = query.toString();
    return encoded ? `?${encoded}` : '';
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<ApiResult<T>> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}${path}`, init, HTTP_TIMEOUT);
        if (!response.ok) {
            const payload = await response.json().catch(() => ({}));
            return { data: null, error: extractErrorMessage(response, payload) };
        }
        return { data: (await response.json()) as T, error: null };
    } catch (error) {
        return {
            data: null,
            error: toErrorMessage(error, 'An unknown public data error occurred.'),
        };
    }
}

export function fetchPublicSources(): Promise<ApiResult<PublicSourceListResponse>> {
    return requestJson('/public-data/sources');
}

export function checkPublicSourceHealth(sourceKey: string): Promise<ApiResult<PublicDataSyncRun>> {
    return requestJson(`/public-data/sources/${encodeURIComponent(sourceKey)}/health-check`, { method: 'POST' });
}

export function fetchNistEditedExperiments(): Promise<ApiResult<NISTEditedExperimentsResponse>> {
    return requestJson('/public-data/sources/nist/edited-experiments');
}

export function fetchNistCollections(): Promise<ApiResult<NISTCollectionListResponse>> {
    return requestJson('/public-data/sources/nist/collections');
}

export function searchNistIsotherms(params: {
    adsorbate?: string;
    adsorbent?: string;
    experiment?: string;
    temperature_min?: number | null;
    temperature_max?: number | null;
    limit?: number;
    offset?: number;
}): Promise<ApiResult<NISTSearchResponse>> {
    return requestJson(`/public-data/sources/nist/search${queryString(params)}`);
}

export function lookupChemicals(q: string): Promise<ApiResult<ChemicalLookupResponse>> {
    return requestJson(`/public-data/chemicals/lookup${queryString({ q })}`);
}

export function fetchNistIsotherms(identifiers: string[]): Promise<ApiResult<PublicDataSyncRun>> {
    return startPublicDataSync({
        source_key: 'nist',
        operation: 'nist_category',
        category: 'experiments',
        identifiers,
    });
}

export function enrichChemicals(fraction: number): Promise<ApiResult<PublicDataSyncRun>> {
    return startPublicDataSync({ source_key: 'pubchem', operation: 'pubchem_enrich', fraction });
}

export function fetchBulkStructures(fraction: number): Promise<ApiResult<PublicDataSyncRun>> {
    return startPublicDataSync({ source_key: 'cod', operation: 'cod_import', fraction });
}

export function startPublicDataSync(request: PublicDataSyncRequest): Promise<ApiResult<PublicDataSyncRun>> {
    return requestJson('/public-data/sync-runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
    });
}

export function fetchPublicDataSyncRun(runId: string): Promise<ApiResult<PublicDataSyncRun>> {
    return requestJson(`/public-data/sync-runs/${encodeURIComponent(runId)}`);
}

export function fetchPublicDataSyncRuns(params: {
    page?: number;
    pageSize?: number;
    scope?: 'active' | 'history';
} = {}): Promise<ApiResult<PublicDataSyncRunListResponse>> {
    const page = Math.max(1, params.page ?? 1);
    const pageSize = Math.min(100, Math.max(1, params.pageSize ?? 20));
    return requestJson(`/public-data/sync-runs${queryString({
        limit: pageSize,
        offset: (page - 1) * pageSize,
        scope: params.scope,
    })}`);
}

export function resumePublicDataSyncRun(runId: string): Promise<ApiResult<PublicDataSyncRun>> {
    return requestJson(`/public-data/sync-runs/${encodeURIComponent(runId)}/resume`, { method: 'POST' });
}

export function fetchPublicDataSyncFailures(runId: string): Promise<ApiResult<PublicDataSyncFailureListResponse>> {
    return requestJson(`/public-data/sync-runs/${encodeURIComponent(runId)}/failures?limit=100&offset=0`);
}

export function cancelPublicDataSyncRun(runId: string): Promise<ApiResult<PublicDataSyncRun>> {
    return requestJson(`/public-data/sync-runs/${encodeURIComponent(runId)}/cancel`, { method: 'POST' });
}

export function fetchPublicOverview(): Promise<ApiResult<PublicDataOverviewResponse>> {
    return requestJson('/public-data/overview');
}

export function resolvePubChem(query: string, adsorbateId?: number | null): Promise<ApiResult<PublicDataSyncRun>> {
    return startPublicDataSync({
        source_key: 'pubchem',
        operation: 'pubchem_resolve',
        query,
        ...(adsorbateId == null ? {} : { adsorbate_id: adsorbateId }),
    });
}

export function searchCOD(params: {
    q?: string;
    formula?: string;
    cod_id?: string;
}): Promise<ApiResult<CODSearchResponse>> {
    return requestJson(`/public-data/structures/search${queryString(params)}`);
}

export function importCOD(codIds: string[], adsorbentId?: number | null): Promise<ApiResult<PublicDataSyncRun>> {
    return startPublicDataSync({
        source_key: 'cod',
        operation: 'cod_import',
        ...(codIds.length === 1 ? { cod_id: codIds[0] } : { cod_ids: codIds }),
        ...(adsorbentId == null ? {} : { adsorbent_id: adsorbentId }),
    });
}

export function fetchPublicStructures(params: {
    page?: number;
    page_size?: number;
    q?: string;
    source?: string;
    linked_only?: boolean | null;
    material_id?: number | null;
}): Promise<ApiResult<StructurePageResponse>> {
    return requestJson(`/public-data/structures${queryString(params)}`);
}

export function fetchPublicStructureDetail(id: number): Promise<ApiResult<StructureRecordView>> {
    return requestJson(`/public-data/structures/${id}`);
}

export function setStructureMaterial(structureId: number, adsorbentId: number | null): Promise<ApiResult<StructureRecordView>> {
    return requestJson(`/public-data/structures/${structureId}/material`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ adsorbent_id: adsorbentId }),
    });
}

export function structureFileUrl(structureId: number): string {
    return `${API_BASE_URL}/public-data/structures/${structureId}/file`;
}
