// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type {
    FittingConfiguration,
    FittingDatasetReadiness,
    FittingSessionDetail,
    FittingSessionListResponse,
    FittingSessionPayload,
    FittingSessionQuery,
    FittingSessionStartResponse,
    ModelCatalogEntry,
    ModelCatalogResponse,
    ModelSettingsUpdate,
} from '../models/fitting.model';
import { API_BASE_URL } from '../core/config/api-base-url';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';
import { toErrorMessage } from '../shared/utils/error-message';

export interface ServiceResult<T> {
    data: T | null;
    error: string | null;
}

async function request<T>(path: string, init: RequestInit = { method: 'GET' }): Promise<ServiceResult<T>> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}${path}`, init, HTTP_TIMEOUT);
        if (response.status === 204) {
            return { data: null, error: null };
        }
        const body = await response.json().catch(() => null);
        if (!response.ok) {
            return { data: null, error: extractErrorMessage(response, body ?? {}) };
        }
        if (!body || typeof body !== 'object') {
            return { data: null, error: 'The server returned an invalid response.' };
        }
        return { data: body as T, error: null };
    } catch (error) {
        return { data: null, error: toErrorMessage(error, 'An unknown error occurred.') };
    }
}

const json = (method: string, payload: unknown): RequestInit => ({
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
});

export const fetchFittingConfiguration = (): Promise<ServiceResult<FittingConfiguration>> =>
    request('/system/configuration');

export const fetchModelCatalog = (): Promise<ServiceResult<ModelCatalogResponse>> =>
    request('/fitting/models');

export const saveModelSettings = (key: string, update: ModelSettingsUpdate): Promise<ServiceResult<ModelCatalogEntry>> =>
    request(`/fitting/models/${encodeURIComponent(key)}`, json('PUT', update));

export const restoreModelDefaults = (key: string): Promise<ServiceResult<ModelCatalogEntry>> =>
    request(`/fitting/models/${encodeURIComponent(key)}/reset`, { method: 'POST' });

export const fetchDatasetReadiness = (datasetId: number): Promise<ServiceResult<FittingDatasetReadiness>> =>
    request(`/fitting/datasets/${datasetId}/readiness`);

export const startFittingSession = (payload: FittingSessionPayload): Promise<ServiceResult<FittingSessionStartResponse>> =>
    request('/fitting/sessions', json('POST', payload));

export const fetchFittingSession = (sessionId: number): Promise<ServiceResult<FittingSessionDetail>> =>
    request(`/fitting/sessions/${sessionId}`);

export const deleteFittingSession = async (sessionId: number): Promise<{ error: string | null }> =>
    ({ error: (await request(`/fitting/sessions/${sessionId}`, { method: 'DELETE' })).error });

export const cancelFittingJob = async (jobId: string): Promise<{ error: string | null }> =>
    ({ error: (await request(`/fitting/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' })).error });

export function fittingSessionParams(query: FittingSessionQuery): URLSearchParams {
    const params = new URLSearchParams({
        sort: query.sort,
        direction: query.direction,
        page: String(query.page),
        page_size: String(query.pageSize),
    });
    if (query.query.trim()) params.set('query', query.query.trim());
    query.statuses.forEach((status) => params.append('status', status));
    if (query.datasetId !== null) params.set('dataset_id', String(query.datasetId));
    if (query.model) params.set('model', query.model);
    if (query.createdFrom) params.set('created_from', query.createdFrom);
    if (query.createdTo) params.set('created_to', query.createdTo);
    return params;
}

export const fetchFittingSessions = (query: FittingSessionQuery): Promise<ServiceResult<FittingSessionListResponse>> =>
    request(`/fitting/sessions?${fittingSessionParams(query).toString()}`);
