export interface DatasetPayload {
    dataset_name: string;
    columns: string[];
    records: Record<string, unknown>[];
}

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export interface JsonObject {
    [key: string]: JsonValue;
}

export type InfoModalValue = JsonValue | undefined;
export type InfoModalData = Record<string, InfoModalValue>;

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

export interface NISTFetchRequest {
    experiments_fraction: number;
    guest_fraction: number;
    host_fraction: number;
}

export interface NISTFetchResponse {
    status: string;
    experiments_count: number;
    single_component_rows: number;
    binary_mixture_rows: number;
    guest_rows: number;
    host_rows: number;
    detail?: string;
    message?: string;
}

export interface NISTPropertiesRequest {
    target: 'guest' | 'host';
}

export interface NISTPropertiesResponse {
    status: string;
    target: string;
    names_requested: number;
    names_matched: number;
    rows_updated: number;
    detail?: string;
    message?: string;
}

export interface NISTStatusResponse {
    status: string;
    data_available: boolean;
    single_component_rows: number;
    binary_mixture_rows: number;
    guest_rows: number;
    host_rows: number;
    detail?: string;
    message?: string;
}

export type NISTCategoryKey = 'experiments' | 'guest' | 'host';

export interface NISTCategoryFetchRequest {
    fraction: number;
}

export interface NISTCategoryRecordStatus {
    category: NISTCategoryKey;
    local_count: number;
    available_count: number;
    last_update: string | null;
    server_ok: boolean | null;
    server_checked_at: string | null;
    supports_enrichment: boolean;
}

export interface NISTCategoryStatusResponse {
    status: string;
    categories: NISTCategoryRecordStatus[];
    detail?: string;
    message?: string;
}

export interface NISTCategoryPingResponse {
    status: string;
    category: NISTCategoryKey;
    server_ok: boolean;
    checked_at: string;
    detail?: string;
    message?: string;
}

export interface NISTCategoryOperationResponse {
    status: string;
    category: NISTCategoryKey;
    available_count?: number;
    local_count?: number;
    requested_count?: number;
    fetched_count?: number;
    names_requested?: number;
    names_matched?: number;
    rows_updated?: number;
    detail?: string;
    message?: string;
}

export interface JobStartResponse {
    job_id: string;
    job_type: string;
    status: string;
    message: string;
    poll_interval?: number;
}

export interface JobStatusResponse {
    job_id: string;
    job_type: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    result?: Record<string, unknown>;
    error?: string;
    poll_interval?: number;
}

export type ParameterKey = [string, string, string];
