import { API_BASE_URL } from '../core/config/api-base-url';
import type { FittingConfiguration } from '../models/fitting.model';
import type { TrainingConfiguration } from '../models/training.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';

export interface ServiceCapability {
    configured: boolean;
    health: string;
    readiness: string;
    version?: string;
    reason?: string;
}

export interface CoreCapabilities {
    configured_mode: 'core' | 'core-ml';
    version: string;
    features: {
        datasets: boolean;
        nist: boolean;
        fitting: boolean;
        training: boolean;
        checkpoints: boolean;
    };
    services: Record<string, ServiceCapability>;
}

export interface ServiceHealth {
    service: string;
    version: string;
    state: string;
}

const asRecord = (value: unknown): Record<string, unknown> | null => (
    value !== null && typeof value === 'object' ? value as Record<string, unknown> : null
);

const readJson = async <T>(url: string): Promise<{ data: T | null; error: string | null }> => {
    try {
        const response = await fetchWithTimeout(url, { method: 'GET' }, HTTP_TIMEOUT);
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
            return { data: null, error: extractErrorMessage(response, body) };
        }
        if (!asRecord(body)) {
            return { data: null, error: `Invalid response from ${url}.` };
        }
        return { data: body as T, error: null };
    } catch (error) {
        return { data: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
};

export const fetchCoreCapabilities = (): Promise<{ data: CoreCapabilities | null; error: string | null }> =>
    readJson<CoreCapabilities>(`${API_BASE_URL}/system/capabilities`);

export const fetchFittingConfiguration = (): Promise<{ data: FittingConfiguration | null; error: string | null }> =>
    readJson<FittingConfiguration>(`${API_BASE_URL}/system/configuration`);

export const fetchTrainingConfiguration = (): Promise<{ data: TrainingConfiguration | null; error: string | null }> =>
    readJson<TrainingConfiguration>(`${API_BASE_URL}/training/configuration`);

export const fetchCoreReadiness = (): Promise<{ data: ServiceHealth | null; error: string | null }> =>
    readJson<ServiceHealth>('/health/ready');

export const fetchMlReadiness = (): Promise<{ data: ServiceHealth | null; error: string | null }> =>
    readJson<ServiceHealth>('/ml-health/ready');
