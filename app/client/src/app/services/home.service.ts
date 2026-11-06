// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { API_BASE_URL } from '../core/config/api-base-url';
import type { HomeSummary } from '../models/home.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';
import { toErrorMessage } from '../shared/utils/error-message';

export async function fetchHomeSummary(): Promise<{ data: HomeSummary | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/home/summary`, { method: 'GET' }, HTTP_TIMEOUT);
        if (!response.ok) {
            const payload = await response.json().catch(() => ({}));
            return { data: null, error: extractErrorMessage(response, payload) };
        }
        return { data: (await response.json()) as HomeSummary, error: null };
    } catch (error) {
        return { data: null, error: toErrorMessage(error, 'Could not load the overview.') };
    }
}
