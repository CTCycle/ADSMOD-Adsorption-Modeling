import type { FittingConfiguration, FittingPayload, FittingResponse, ModelCatalogResponse } from '../models/fitting.model';
import { API_BASE_URL } from '../core/config/api-base-url';
import type { JobStatusResponse } from '../models/job.model';
import { pollJobUntilTerminal, startJob } from './job.service';
import type { JobStartResult } from './job.service';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';

export async function startFittingJob(
    payload: FittingPayload
): Promise<JobStartResult> {
    return startJob('/fitting/run', payload);
}

export async function pollFittingJobUntilComplete(
    jobId: string,
    pollInterval?: number,
    onProgress?: (status: JobStatusResponse) => void
): Promise<{ message: string; data: FittingResponse | null }> {
    const result = await pollJobUntilTerminal('/fitting', jobId, pollInterval, onProgress);
    if (result.error || !result.data) {
        return { message: `[ERROR] ${result.error || 'Job status response was empty.'}`, data: null };
    }
    const status = result.data;

    if (status.status === 'completed') {
        const result = status.result as FittingResponse | undefined;
        if (!result) {
            return { message: '[INFO] Fitting completed.', data: null };
        }

        const defaultMessage =
            result.status === 'error'
                ? '[ERROR] Fitting finished without any successful model fits.'
                : result.status === 'warning'
                    ? '[WARN] Fitting completed with partial issues.'
                    : '[INFO] Fitting completed successfully.';
        const summary = result.summary?.trim();
        if (!summary) {
            return { message: defaultMessage, data: result };
        }
        const prefix = result.status === 'error'
            ? '[ERROR]'
            : result.status === 'warning'
                ? '[WARN]'
                : '[INFO]';
        return { message: `${prefix} ${summary}`, data: result };
    }

    if (status.status === 'failed') {
        return { message: `[ERROR] ${status.error || 'Job failed.'}`, data: null };
    }

    return { message: '[INFO] Job was cancelled.', data: null };
}

export async function fetchModelCatalog(pressure = 'bar', uptake = 'mmol/g'): Promise<{ data: ModelCatalogResponse | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/fitting/models?pressure_unit=${encodeURIComponent(pressure)}&uptake_unit=${encodeURIComponent(uptake)}`,
            { method: 'GET' },
            HTTP_TIMEOUT,
        );
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
            return { data: null, error: extractErrorMessage(response, body) };
        }
        if (!body || typeof body !== 'object') {
            return { data: null, error: 'Invalid model catalogue response.' };
        }
        return { data: body as ModelCatalogResponse, error: null };
    } catch (error) {
        return { data: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}

export async function fetchFittingConfiguration(): Promise<{ data: FittingConfiguration | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/system/configuration`, { method: 'GET' }, HTTP_TIMEOUT);
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
            return { data: null, error: extractErrorMessage(response, body) };
        }
        if (!body || typeof body !== 'object') {
            return { data: null, error: 'Invalid fitting configuration response.' };
        }
        return { data: body as FittingConfiguration, error: null };
    } catch (error) {
        return { data: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}
