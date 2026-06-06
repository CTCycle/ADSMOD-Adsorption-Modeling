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
