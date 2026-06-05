export interface DatasetPayload {
    dataset_name: string;
    columns: string[];
    records: Record<string, unknown>[];
}

export interface DatasetResponse {
    status: string;
    dataset?: DatasetPayload;
    summary?: string;
    detail?: string;
    message?: string;
}
