// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { API_BASE_URL } from '../core/config/api-base-url';
import type { MaterialDetail, MaterialPage, SpeciesDetail, SpeciesPage } from '../models/materials.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';
import { toErrorMessage } from '../shared/utils/error-message';

export interface ApiResult<T> {
    data: T | null;
    error: string | null;
}

export interface EntityListParams {
    page?: number;
    page_size?: number;
    q?: string;
    source?: string | null;
}

function queryString(params: EntityListParams): string {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null && value !== '') {
            query.set(key, String(value));
        }
    }
    const encoded = query.toString();
    return encoded ? `?${encoded}` : '';
}

async function request<T>(path: string, init: RequestInit = {}): Promise<ApiResult<T>> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}${path}`, init, HTTP_TIMEOUT);
        const payload: unknown = response.status === 204 ? null : await response.json().catch(() => ({}));
        return response.ok
            ? { data: payload as T, error: null }
            : { data: null, error: extractErrorMessage(response, payload) };
    } catch (error) {
        return { data: null, error: toErrorMessage(error, 'Unable to reach the backend.') };
    }
}

export function fetchMaterials(params: EntityListParams = {}): Promise<ApiResult<MaterialPage>> {
    return request(`/materials${queryString(params)}`);
}

export function fetchMaterialDetail(id: number): Promise<ApiResult<MaterialDetail>> {
    return request(`/materials/${id}`);
}

export function deleteMaterial(id: number): Promise<ApiResult<null>> {
    return request(`/materials/${id}`, { method: 'DELETE' });
}

export function fetchSpecies(params: EntityListParams = {}): Promise<ApiResult<SpeciesPage>> {
    return request(`/species${queryString(params)}`);
}

export function fetchSpeciesDetail(id: number): Promise<ApiResult<SpeciesDetail>> {
    return request(`/species/${id}`);
}

export function deleteSpecies(id: number): Promise<ApiResult<null>> {
    return request(`/species/${id}`, { method: 'DELETE' });
}
