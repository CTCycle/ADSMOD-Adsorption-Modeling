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

export const API_BASE_URL = '/api';
