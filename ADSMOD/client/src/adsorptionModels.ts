// Data model for the 9 adsorption models
// This file contains model metadata including descriptions and LaTeX equations

export interface ModelConfigField {
    name: string;
    label: string;
    type: 'number' | 'select' | 'boolean' | 'text';
    min?: number;
    max?: number;
    step?: number;
    defaultValue?: number | string | boolean;
    options?: { value: string; label: string }[];
}

export interface AdsorptionModel {
    id: string;
    name: string;
    shortDescription: string;
    equationLatex: string;
    configSchema?: ModelConfigField[];
    parameterDefaults: Record<string, [number, number]>;
}

export const ADSORPTION_MODELS: AdsorptionModel[] = [
    {
        id: 'langmuir',
        name: 'Langmuir',
        shortDescription:
            'Monolayer adsorption on homogeneous surfaces with identical sites and finite saturation capacity.',
        equationLatex: 'q_e = \\frac{q_{\\max} K_L C_e}{1 + K_L C_e}',
        parameterDefaults: {
            k: [1e-6, 10.0],
            qsat: [0.0, 100.0],
        },
    },
    {
        id: 'freundlich',
        name: 'Freundlich',
        shortDescription:
            'Empirical power-law isotherm for heterogeneous surfaces without explicit saturation.',
        equationLatex: 'q_e = K_F C_e^{1/n_F}',
        parameterDefaults: {
            k: [1e-6, 10.0],
            exponent: [0.1, 10.0],
        },
    },
    {
        id: 'sips',
        name: 'Sips (Langmuir–Freundlich)',
        shortDescription:
            'Combines Langmuir and Freundlich behavior for heterogeneous surfaces with saturation.',
        equationLatex: 'q_e = \\frac{q_{\\max} K_S C_e^{n_S}}{1 + K_S C_e^{n_S}}',
        parameterDefaults: {
            k: [1e-6, 10.0],
            qsat: [0.0, 100.0],
            exponent: [0.1, 10.0],
        },
    },
    {
        id: 'temkin',
        name: 'Temkin',
        shortDescription:
            'Accounts for adsorbate interactions with heat of adsorption decreasing linearly with coverage.',
        equationLatex: 'q_e = B_T \\ln\\bigl(K_T C_e\\bigr)',
        parameterDefaults: {
            k: [1e-6, 10.0],
            beta: [0.1, 10.0],
        },
    },
    {
        id: 'toth',
        name: 'Toth',
        shortDescription:
            'Modified Langmuir for heterogeneous surfaces with improved fit at high loading.',
        equationLatex: 'q_e = \\frac{q_{\\max} K_T C_e}{\\left(1 + (K_T C_e)^{n_T}\\right)^{1/n_T}}',
        parameterDefaults: {
            k: [1e-6, 10.0],
            qsat: [0.0, 100.0],
            exponent: [0.1, 10.0],
        },
    },
    {
        id: 'dubinin_radushkevich',
        name: 'Dubinin–Radushkevich',
        shortDescription:
            'Micropore filling model based on Polanyi potential theory for porous solids.',
        equationLatex: 'q_e = q_{\\max} \\exp\\bigl(-K_{DR} \\varepsilon^{2}\\bigr)',
        parameterDefaults: {
            qsat: [0.0, 100.0],
            beta: [1e-6, 10.0],
        },
    },
    {
        id: 'dual_site_langmuir',
        name: 'Dual Site Langmuir',
        shortDescription:
            'Two independent site families with separate capacities; total loading is their sum.',
        equationLatex:
            'q_e = \\frac{q_{1,\\max} K_1 C_e}{1 + K_1 C_e} + \\frac{q_{2,\\max} K_2 C_e}{1 + K_2 C_e}',
        parameterDefaults: {
            k1: [1e-6, 10.0],
            qsat1: [0.0, 100.0],
            k2: [1e-6, 10.0],
            qsat2: [0.0, 100.0],
        },
    },
    {
        id: 'redlich_peterson',
        name: 'Redlich–Peterson',
        shortDescription:
            'Empirical model interpolating between Langmuir and Freundlich over broad ranges.',
        equationLatex: 'q_e = \\frac{K_{RP} C_e}{1 + a_{RP} C_e^{\\beta}}',
        parameterDefaults: {
            k: [1e-6, 10.0],
            a: [1e-6, 10.0],
            beta: [0.1, 1.0],
        },
    },
    {
        id: 'jovanovic',
        name: 'Jovanovic',
        shortDescription:
            'Monolayer isotherm with exponential approach to saturation, suited for rigid adsorbents.',
        equationLatex: 'q_e = q_{\\max} \\bigl(1 - \\exp(-K_J C_e)\\bigr)',
        parameterDefaults: {
            k: [1e-6, 10.0],
            qsat: [0.0, 100.0],
        },
    },
];

// Mapping from model id to display name for backward compatibility
export const MODEL_ID_TO_NAME: Record<string, string> = {
    langmuir: 'Langmuir',
    freundlich: 'Freundlich',
    sips: 'Sips',
    temkin: 'Temkin',
    toth: 'Toth',
    dubinin_radushkevich: 'Dubinin-Radushkevich',
    dual_site_langmuir: 'Dual-Site Langmuir',
    redlich_peterson: 'Redlich-Peterson',
    jovanovic: 'Jovanovic',
};

// Mapping from display name to model id
export const MODEL_NAME_TO_ID: Record<string, string> = Object.entries(MODEL_ID_TO_NAME).reduce(
    (acc, [id, name]) => {
        acc[name] = id;
        return acc;
    },
    {} as Record<string, string>
);
