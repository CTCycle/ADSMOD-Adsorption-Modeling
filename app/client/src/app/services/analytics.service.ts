// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { API_BASE_URL } from '../core/config/api-base-url';
import type {
    AnalyticsOverviewResponse,
    DistributionRequest,
    DistributionResponse,
    SelectionRequest,
    SelectionResponse,
    SeriesRequest,
    SeriesResponse,
    UptakeAtPressureRequest,
    UptakeAtPressureResponse,
} from '../models/analytics.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';
import type { ApiResult } from './public-data.service';

async function requestJson<T>(path: string, body?: unknown): Promise<ApiResult<T>> {
    const init: RequestInit = body === undefined
        ? {}
        : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/analytics${path}`, init, HTTP_TIMEOUT);
        if (!response.ok) {
            const payload = await response.json().catch(() => ({}));
            return { data: null, error: extractErrorMessage(response, payload) };
        }
        return { data: (await response.json()) as T, error: null };
    } catch (error) {
        return {
            data: null,
            error: error instanceof Error ? error.message : 'An unknown analytics error occurred.',
        };
    }
}

export function fetchAnalyticsOverview(): Promise<ApiResult<AnalyticsOverviewResponse>> {
    return requestJson('/overview');
}

export function queryAnalyticsSelection(request: SelectionRequest): Promise<ApiResult<SelectionResponse>> {
    return requestJson('/selection', request);
}

export function queryAnalyticsSeries(request: SeriesRequest): Promise<ApiResult<SeriesResponse>> {
    return requestJson('/series', request);
}

export function queryUptakeAtPressure(request: UptakeAtPressureRequest): Promise<ApiResult<UptakeAtPressureResponse>> {
    return requestJson('/uptake-at-pressure', request);
}

export function queryAnalyticsDistribution(request: DistributionRequest): Promise<ApiResult<DistributionResponse>> {
    return requestJson('/distribution', request);
}
