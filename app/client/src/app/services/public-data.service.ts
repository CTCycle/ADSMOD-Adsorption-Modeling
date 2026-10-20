// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { API_BASE_URL } from '../core/config/api-base-url';
import type {
    AdsorptionDetailResponse,
    AdsorptionPageResponse,
    CODSearchResponse,
    ChemicalPageResponse,
    ChemicalRecordView,
    MaterialDetailResponse,
    MaterialPageResponse,
    NISTCollectionListResponse,
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
            error: error instanceof Error ? error.message : 'An unknown public data error occurred.',
        };
    }
}

export function fetchPublicSources(): Promise<ApiResult<PublicSourceListResponse>> {
    return requestJson('/public-data/sources');
}

export function checkPublicSourceHealth(sourceKey: string): Promise<ApiResult<PublicDataSyncRun>> {
    return requestJson(`/public-data/sources/${encodeURIComponent(sourceKey)}/health-check`, { method: 'POST' });
}

export function fetchNistCollections(): Promise<ApiResult<NISTCollectionListResponse>> {
    return requestJson('/public-data/sources/nist/collections');
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

export function fetchPublicDataSyncRuns(): Promise<ApiResult<PublicDataSyncRunListResponse>> {
    return requestJson('/public-data/sync-runs?limit=20&offset=0');
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

export function fetchPublicAdsorption(params: {
    page?: number;
    page_size?: number;
    source?: string;
    material?: string;
    adsorbate?: string;
    material_id?: number | null;
    adsorbate_id?: number | null;
    temperature_min_k?: number | null;
    temperature_max_k?: number | null;
}): Promise<ApiResult<AdsorptionPageResponse>> {
    return requestJson(`/public-data/adsorption${queryString(params)}`);
}

export function fetchPublicAdsorptionDetail(id: number): Promise<ApiResult<AdsorptionDetailResponse>> {
    return requestJson(`/public-data/adsorption/${id}`);
}

export function fetchPublicMaterials(params: {
    page?: number;
    page_size?: number;
    q?: string;
    formula?: string;
    source?: string;
    has_structure?: boolean | null;
}): Promise<ApiResult<MaterialPageResponse>> {
    return requestJson(`/public-data/materials${queryString(params)}`);
}

export function fetchPublicMaterialDetail(id: number): Promise<ApiResult<MaterialDetailResponse>> {
    return requestJson(`/public-data/materials/${id}`);
}

export function fetchPublicChemicals(params: {
    page?: number;
    page_size?: number;
    q?: string;
    formula?: string;
    source?: string;
    molecular_weight_min?: number | null;
    molecular_weight_max?: number | null;
}): Promise<ApiResult<ChemicalPageResponse>> {
    return requestJson(`/public-data/chemicals${queryString(params)}`);
}

export function fetchPublicChemicalDetail(id: number): Promise<ApiResult<ChemicalRecordView>> {
    return requestJson(`/public-data/chemicals/${id}`);
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
