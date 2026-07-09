export function normalizeApiBaseUrl(rawValue: string): string {
    const trimmed = rawValue.trim();
    if (!trimmed) {
        return '/api';
    }
    if (/^https?:\/\//i.test(trimmed) || trimmed.startsWith('//')) {
        return '/api';
    }

    const withLeadingSlash = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
    if (!/^\/[A-Za-z0-9/_-]*$/.test(withLeadingSlash)) {
        return '/api';
    }

    return withLeadingSlash.replace(/\/+$/, '') || '/api';
}

const importMetaEnv = (import.meta as unknown as { env?: { VITE_API_BASE_URL?: string } }).env;
const apiBaseEnv = importMetaEnv?.VITE_API_BASE_URL || '';
export const API_BASE_URL = normalizeApiBaseUrl(apiBaseEnv);
