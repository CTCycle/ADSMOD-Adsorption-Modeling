// Type definitions for ADSMOD frontend

export interface DatasetPayload {
    columns: string[];
    records: Record<string, unknown>[];
}

export interface ParameterBound {
    min: number;
    max: number;
}

export interface ModelParameters {
    [parameterName: string]: ParameterBound;
}

export interface ModelConfiguration {
    min: Record<string, number>;
    max: Record<string, number>;
    initial: Record<string, number>;
}

export interface FittingPayload {
    max_iterations: number;
    optimization_method: 'LSS' | 'BFGS' | 'L-BFGS-B' | 'Nelder-Mead' | 'Powell';
    parameter_bounds: Record<string, ModelConfiguration>;
    dataset: DatasetPayload;
}

export interface DatasetResponse {
    status: string;
    dataset?: DatasetPayload;
    summary?: string;
    detail?: string;
    message?: string;
}

export interface FittingResponse {
    status: string;
    summary?: string;
    detail?: string;
    message?: string;
    processed_rows?: number;
    best_model_saved?: boolean;
    models?: string[];
}

export type ParameterKey = [string, string, string]; // [model, parameter, bound_type]

// Browser API types
export interface TableInfo {
    table_name: string;
    display_name: string;
}

export interface TableListResponse {
    status: string;
    tables: TableInfo[];
}

export interface TableDataResponse {
    status: string;
    table_name: string;
    display_name: string;
    row_count: number;
    column_count: number;
    columns: string[];
    data: Record<string, unknown>[];
}
