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

const apiBaseEnv = (import.meta.env.VITE_API_BASE_URL || "").trim();
export const API_BASE_URL = (apiBaseEnv || "/api").replace(/\/+$/, "");
