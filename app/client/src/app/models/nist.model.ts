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
