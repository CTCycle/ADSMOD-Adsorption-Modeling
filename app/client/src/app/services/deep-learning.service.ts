// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { API_BASE_URL } from '../core/config/api-base-url';
import type { DeepLearningAction, DeepLearningPlan, DeepLearningStatus } from '../models/deep-learning.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';

const DEEP_LEARNING_URL = `${API_BASE_URL}/deep-learning`;
const RESTART_POLL_MS = 3000;
const RESTART_STOP_GRACE_MS = 90_000;
const RESTART_TIMEOUT_MS = 45 * 60_000;

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    if (init.body !== undefined) headers.set('Content-Type', 'application/json');
    const response = await fetchWithTimeout(`${DEEP_LEARNING_URL}${path}`, { ...init, headers }, HTTP_TIMEOUT);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(extractErrorMessage(response, body));
    return body as T;
}

export const fetchDeepLearningStatus = (refresh = false): Promise<DeepLearningStatus> =>
    request<DeepLearningStatus>(`/status${refresh ? '?refresh=true' : ''}`);

export const fetchDeepLearningPlan = (action: DeepLearningAction): Promise<DeepLearningPlan> =>
    request<DeepLearningPlan>(`/plan?action=${action}`);

export const startDeepLearningOperation = (action: DeepLearningAction): Promise<{ operation_id: string; action: DeepLearningAction }> =>
    request('/operations', { method: 'POST', body: JSON.stringify({ action, confirm: true }) });

export const fetchDeepLearningLog = async (lines = 80): Promise<string[]> =>
    (await request<{ lines: string[] }>(`/log?lines=${lines}`)).lines;

const isBackendLive = async (): Promise<boolean> => {
    try {
        return (await fetchWithTimeout('/health/live', { method: 'GET', cache: 'no-store' }, 2500)).ok;
    } catch {
        return false;
    }
};

/**
 * Resolves once the backend has gone away and come back (the managed restart).
 * If it never goes away within the grace period the caller still reloads, which
 * surfaces the real status (for example a rejected operation).
 */
export async function waitForBackendRestart(sleep: (ms: number) => Promise<void> = (ms) => new Promise((resolve) => setTimeout(resolve, ms))): Promise<void> {
    const started = Date.now();
    let wentDown = false;
    while (Date.now() - started < RESTART_TIMEOUT_MS) {
        const live = await isBackendLive();
        if (!live) wentDown = true;
        else if (wentDown) return;
        else if (Date.now() - started > RESTART_STOP_GRACE_MS) return;
        await sleep(RESTART_POLL_MS);
    }
}
