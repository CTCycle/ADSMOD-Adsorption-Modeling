import { API_BASE_URL } from '../core/config/api-base-url';
import type {
    DatabaseSettings,
    DatabaseSettingsRequest,
    DatabaseTestResponse,
    SettingsResponse,
    UserSettings,
} from '../models/settings.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';

async function requestSettings(
    url: string,
    init: RequestInit = {},
): Promise<SettingsResponse> {
    const headers = new Headers(init.headers);
    if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
    const response = await fetchWithTimeout(
        url,
        { ...init, headers },
        HTTP_TIMEOUT,
    );
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(extractErrorMessage(response, body));
    }
    return body as SettingsResponse;
}

const SETTINGS_URL = `${API_BASE_URL}/settings`;

export const fetchSettings = (): Promise<SettingsResponse> => requestSettings(SETTINGS_URL);

export const saveSettings = (settings: UserSettings): Promise<SettingsResponse> =>
    requestSettings(SETTINGS_URL, { method: 'PUT', body: JSON.stringify(settings) });

export const resetSettings = (): Promise<SettingsResponse> =>
    requestSettings(`${SETTINGS_URL}/reset`, { method: 'POST' });

async function requestDatabase<T>(url: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
    const response = await fetchWithTimeout(url, { ...init, headers }, HTTP_TIMEOUT);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(extractErrorMessage(response, body));
    return body as T;
}

const DATABASE_URL = `${API_BASE_URL}/database`;

export const fetchDatabaseSettings = (): Promise<DatabaseSettings> =>
    requestDatabase<DatabaseSettings>(DATABASE_URL);

export const testDatabaseConnection = (settings: DatabaseSettingsRequest): Promise<DatabaseTestResponse> =>
    requestDatabase<DatabaseTestResponse>(`${DATABASE_URL}/test`, {
        method: 'POST',
        body: JSON.stringify(settings),
    });

export const switchDatabase = (settings: DatabaseSettingsRequest): Promise<DatabaseSettings> =>
    requestDatabase<DatabaseSettings>(`${DATABASE_URL}/switch`, {
        method: 'POST',
        body: JSON.stringify(settings),
    });
