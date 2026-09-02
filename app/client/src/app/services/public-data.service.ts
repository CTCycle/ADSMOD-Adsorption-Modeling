import { API_BASE_URL } from '../core/config/api-base-url';
import type {
    AdsorptionDetailResponse,
    AdsorptionPageResponse,
    CODSearchResponse,
    ChemicalPageResponse,
    ChemicalRecordView,
    MaterialPageResponse,
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

export function fetchPublicSources(checkHealth = true): Promise<ApiResult<PublicSourceListResponse>> {
    return requestJson(`/public-data/sources${queryString({ check_health: checkHealth })}`);
}

export function fetchPublicAdsorption(params: {
    page?: number;
    page_size?: number;
    source?: string;
    material?: string;
    adsorbate?: string;
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

export function resolvePubChem(query: string): Promise<ApiResult<ChemicalRecordView>> {
    return requestJson('/public-data/chemicals/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
    });
}

export function searchCOD(params: {
    q?: string;
    formula?: string;
    cod_id?: string;
}): Promise<ApiResult<CODSearchResponse>> {
    return requestJson(`/public-data/structures/search${queryString(params)}`);
}

export function importCOD(codId: string, adsorbentId?: number | null): Promise<ApiResult<StructureRecordView>> {
    return requestJson('/public-data/structures/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cod_id: codId, adsorbent_id: adsorbentId ?? null }),
    });
}

export function fetchPublicStructures(params: {
    page?: number;
    page_size?: number;
    q?: string;
    source?: string;
    linked_only?: boolean | null;
}): Promise<ApiResult<StructurePageResponse>> {
    return requestJson(`/public-data/structures${queryString(params)}`);
}

export function fetchPublicStructureDetail(id: number): Promise<ApiResult<StructureRecordView>> {
    return requestJson(`/public-data/structures/${id}`);
}
