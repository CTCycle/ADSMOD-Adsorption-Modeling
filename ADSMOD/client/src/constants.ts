// Model parameter defaults - canonical names aligned with the Python backend

export const MODEL_PARAMETER_DEFAULTS: Record<string, Record<string, [number, number]>> = {
    Langmuir: {
        k: [1e-6, 10.0],
        qsat: [0.0, 100.0],
    },
    Sips: {
        k: [1e-6, 10.0],
        qsat: [0.0, 100.0],
        exponent: [0.1, 10.0],
    },
    Freundlich: {
        k: [1e-6, 10.0],
        exponent: [0.1, 10.0],
    },
    Temkin: {
        k: [1e-6, 10.0],
        beta: [0.1, 10.0],
    },
    Toth: {
        k: [1e-6, 10.0],
        qsat: [0.0, 100.0],
        exponent: [0.1, 10.0],
    },
    "Dubinin-Radushkevich": {
        qsat: [0.0, 100.0],
        beta: [1e-6, 10.0],
    },
    "Dual-Site Langmuir": {
        k1: [1e-6, 10.0],
        qsat1: [0.0, 100.0],
        k2: [1e-6, 10.0],
        qsat2: [0.0, 100.0],
    },
    "Redlich-Peterson": {
        k: [1e-6, 10.0],
        a: [1e-6, 10.0],
        beta: [0.1, 1.0],
    },
    Jovanovic: {
        k: [1e-6, 10.0],
        qsat: [0.0, 100.0],
    },
};

const API_BASE_FALLBACK = '/api';
const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1']);

function normalizePathApiBase(rawValue: string): string {
    const trimmed = rawValue.trim();
    if (!trimmed) {
        return API_BASE_FALLBACK;
    }

    const withLeadingSlash = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
    if (!/^\/[A-Za-z0-9/_-]*$/.test(withLeadingSlash)) {
        return API_BASE_FALLBACK;
    }

    return withLeadingSlash.replace(/\/+$/, '') || API_BASE_FALLBACK;
}

function normalizeDesktopApiOrigin(rawValue: string): string | null {
    const trimmed = rawValue.trim();
    if (!trimmed) {
        return null;
    }

    try {
        const parsed = new URL(trimmed);
        const protocol = parsed.protocol.toLowerCase();
        if (protocol !== 'http:' && protocol !== 'https:') {
            return null;
        }
        if (!LOOPBACK_HOSTS.has(parsed.hostname.toLowerCase())) {
            return null;
        }

        return parsed.href.replace(/\/+$/, '');
    } catch {
        return null;
    }
}

export function resolveApiBaseUrl(rawValue: string): string {
    return normalizeDesktopApiOrigin(rawValue) || normalizePathApiBase(rawValue);
}

const apiBaseEnv = import.meta.env.VITE_API_BASE_URL || '';

export let API_BASE_URL = resolveApiBaseUrl(apiBaseEnv);

export function setApiBaseUrl(rawValue: string | null | undefined): void {
    if (!rawValue) {
        return;
    }
    API_BASE_URL = resolveApiBaseUrl(rawValue);
}
